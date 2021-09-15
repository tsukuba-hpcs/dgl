/*!
 *  Copyright (c) 2021 by Contributors
 * \file ucx_communicator.cc
 * \brief UCXCommunicator for DGL distributed training.
 */

#include <dmlc/logging.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <utility>

#include "ucx_communicator.h"


namespace dgl {
namespace network {

/////////////////////////////////////// UCXSender ///////////////////////////////////////////

UCXSender::UCXSender(int64_t queue_size) :
  Sender(queue_size),
  prog_queue_(std::make_shared<CircularBuffer<ucs_status_ptr_t>>(queue_size)),
  prog_producer_(std::make_unique<CircularBufferProducer<ucs_status_ptr_t>>(prog_queue_)) {
}

void UCXSender::AddReceiver(const char *addr, int recv_id) {
  CHECK_NOTNULL(addr);
  if (recv_id < 0) {
    LOG(FATAL) << "recv_id cannot be a negative number.";
  }
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // check address format
  if (substring[0] != "ucx:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'ucx://127.0.0.1:50051'. ";
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'ucx://127.0.0.1:50051'. ";
  }
  IPAddr address;
  address.ip = ip_and_port[0];
  address.port = std::stoi(ip_and_port[1]);
  receiver_addrs_[recv_id] = address;
}

bool UCXSender::Connect() {
  ucs_status_t status;
  ucp_params_t params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES,
    .features = UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_AMO32 | UCP_FEATURE_STREAM,
  };
  // Init
  status = ucp_init(&params, NULL, &context_);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_init failed: " << ucs_status_string(status);
    return false;
  }
  // Create Worker
  ucp_worker_params wparams = {
    .field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE,
    .thread_mode = UCS_THREAD_MODE_MULTI,
  };
  status = ucp_worker_create(context_, &wparams, &worker_);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_worker_create failed: " << ucs_status_string(status);
    return false;
  }
  // Create Endpoints
  for (const auto &r : receiver_addrs_) {
    int ID = r.first;
    struct sockaddr_in sockaddr = {
      .sin_family = AF_INET,
      .sin_port = htons(receiver_addrs_[ID].port),
      .sin_addr = {
        .s_addr = inet_addr(receiver_addrs_[ID].ip.c_str()),
      },
    };
    ucp_ep_params_t epparams = {
      .field_mask = UCP_EP_PARAM_FIELD_SOCK_ADDR | UCP_EP_PARAM_FIELD_FLAGS,
      .flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER,
      .sockaddr = {
        .addr = (struct sockaddr*)&sockaddr,
        .addrlen = sizeof(sockaddr),
      },
    };
    status = ucp_ep_create(worker_, &epparams, &eps_[ID]);
    if (status != UCS_OK) {
      LOG(FATAL) << "ucp_ep_create failed: " << ucs_status_string(status)
      << " for " << receiver_addrs_[ID].ip << ":" << receiver_addrs_[ID].port;
      return false;
    }
  }
  // Fork progress thread
  prog_thread_ = std::make_shared<std::thread>(SendLoop, this);
  return true;
}

STATUS UCXSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  ucp_request_param_t rparam = {
    .op_attr_mask = 0,
  };
  ucs_status_ptr_t size_req, data_req;
  size_req = ucp_stream_send_nbx(eps_[recv_id], &msg.size, sizeof(int64_t), &rparam);
  if (UCS_PTR_IS_ERR(size_req)) {
    LOG(FATAL) << "ucp_stream_send_nbx(size_req) failed: " <<
    ucs_status_string(ucp_request_check_status(size_req));
    return QUEUE_CLOSE;
  }
  CHECK_NOTNULL(size_req);
  data_req = ucp_stream_send_nbx(eps_[recv_id], &msg.data, msg.size, &rparam);
  if (UCS_PTR_IS_ERR(data_req)) {
    LOG(FATAL) << "ucp_stream_send_nbx(data_req) failed: " <<
    ucs_status_string(ucp_request_check_status(data_req));
    return QUEUE_CLOSE;
  }
  CHECK_NOTNULL(data_req);
  if (prog_producer_->fill()) {
    LOG(FATAL) << "progress queue is filled";
  }
  prog_producer_->push(std::move(size_req));

  if (prog_producer_->fill()) {
    LOG(FATAL) << "progress queue is filled";
  }
  prog_producer_->push(std::move(data_req));
}


void UCXSender::Finalize() {
  // Send Finish Signal
  if (prog_producer_->fill()) {
    LOG(FATAL) << "progress queue is filled";
  }
  prog_producer_->push(NULL);
  // join prog_thread_
  prog_thread_->join();
  // Close endpoints
  for (auto &ep : eps_) {
    ucp_ep_close_nb(ep.second, UCP_EP_CLOSE_MODE_FORCE);
  }
  ucp_worker_destroy(worker_);
  ucp_cleanup(context_);
}

void UCXSender::SendLoop(UCXSender *self) {
  CHECK_NOTNULL(self);
  ucs_status_t status;
  CircularBufferConsumer<ucs_status_ptr_t> consumer(self->prog_queue_);
  while (true) {
    ucp_worker_progress(self->worker_);
    {
      if (consumer.empty()) {
        continue;
      }
      ucs_status_ptr_t stat = *consumer.front();
      if (stat == NULL) {
        std::cout << "Finish Signal is received";
        return;
      }
      status = ucp_request_check_status(stat);
      if (status == UCS_INPROGRESS) {
        continue;
      }
      if (UCS_PTR_IS_ERR(stat)) {
        LOG(FATAL) << "sendloop: ucp_stream_send failed " <<
        ucs_status_string(status);
      }
      ucp_request_free(stat);
      consumer.pop();
    }
  }
}

/////////////////////////////////////// UCXReceiver ///////////////////////////////////////////

UCXReceiver::UCXReceiver(int64_t queue_size) :
  Receiver(queue_size),
  queue_size(queue_size),
  notify_queue_(std::make_shared<CircularBuffer<int>>(queue_size)),
  notify_consumer_(std::make_unique<CircularBufferConsumer<int>>(notify_queue_)) {
}

bool UCXReceiver::Wait(const char* addr, int num_sender) {
  CHECK_NOTNULL(addr);
  if (num_sender <= 0) {
    LOG(FATAL) << "num_sender must be positive number.";
    return false;
  }
  // Get self address
  std::vector<std::string> substring;
  std::vector<std::string> ip_and_port;
  SplitStringUsing(addr, "//", &substring);
  // check address format
  if (substring[0] != "ucx:" || substring.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'ucx://127.0.0.1:50051'. ";
    return false;
  }
  // Get IP and port
  SplitStringUsing(substring[1], ":", &ip_and_port);
  if (ip_and_port.size() != 2) {
    LOG(FATAL) << "Incorrect address format:" << addr
               << " Please provide right address format, "
               << "e.g, 'ucx://127.0.0.1:50051'. ";
    return false;
  }
  IPAddr address;
  address.ip = ip_and_port[0];
  address.port = std::stoi(ip_and_port[1]);

  ucs_status_t status;
  ucp_params_t params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES,
    .features = UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_AMO32 | UCP_FEATURE_STREAM,
  };
  // Init
  status = ucp_init(&params, NULL, &context_);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_init failed: " << ucs_status_string(status);
    return false;
  }
  // Create Worker
  ucp_worker_params wparams = {
    .field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE,
    .thread_mode = UCS_THREAD_MODE_MULTI,
  };
  status = ucp_worker_create(context_, &wparams, &worker_);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_worker_create failed: " << ucs_status_string(status);
    return false;
  }
  // Create Listener
  struct sockaddr_in sockaddr = {
      .sin_family = AF_INET,
      .sin_port = htons(address.port),
  };
  if (inet_aton(address.ip.c_str(), &sockaddr.sin_addr) < 0) {
    LOG(FATAL) << "inet_aton failed: ";
    return false;
  }
  ucp_listener_params_t lparams = {
    .field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
    .sockaddr = {
      .addr = (const struct sockaddr*)&sockaddr,
      .addrlen = sizeof(sockaddr),
    },
    .conn_handler = {
      .arg = this,
      .cb = ConnHandlerCallback,
    }
  };
  status = ucp_listener_create(worker_, &lparams, &listener_);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_listener_create failed: " << ucs_status_string(status);
    return false;
  }
  // Clear eps_
  eps_.clear();
  // Wait until that (number of sender) == num_sender
  while (eps_.size() < num_sender) {
    ucp_worker_progress(worker_);
  }
  // Close listener
  ucp_listener_destroy(listener_);
  // Init receive queues and consumers
  recv_queues_.assign(num_sender, std::make_shared<CircularBuffer<Message>>(queue_size));
  recv_consumer_.clear();
  for (int id=0; id < num_sender; id++) {
    recv_consumer_.emplace_back(
      std::make_unique<CircularBufferConsumer<int>>(recv_queues_[id]));
  }
  // Fork progress thread
  prog_thread_ = std::make_shared<std::thread>(RecvLoop, this);
  return true;
}

void UCXReceiver::ConnHandlerCallback(ucp_conn_request_h conn_request, void *arg) {
  UCXReceiver *self = reinterpret_cast<UCXReceiver *>(arg);
  ucs_status_t status;
  ucp_conn_request_attr_t attr = {
    .field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR,
  };
  status = ucp_conn_request_query(conn_request, &attr);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_conn_request_query failed: " << ucs_status_string(status);
    return;
  }
  ucp_ep_params_t epparams = {
    .field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST,
    .conn_request = conn_request,
  };
  int ID = self->eps_.size();
  self->eps_.emplace_back();
  status = ucp_ep_create(self->worker_, &epparams, &self->eps_[ID]);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_conn_request_query failed: " << ucs_status_string(status);
    self->eps_.pop_back();
    return;
  }
}

std::vector<Message> RecvMsgLoop(UCXStreamBuffer *buf, size_t length, ucs_status_ptr_t data) {
  std::vector<Message> msgs;
  while (length > 0) {
    // If size_buffer is not filled.
    if (buf->size_offset < sizeof(int64_t)) {
      // If size_buffer will be not filled in this iter, finish.
      if (buf->size_offset + length < sizeof(int64_t)) {
        memcpy(
          buf->size_buffer + buf->size_offset,
          data, length);
        buf->size_offset += length;
        return msgs;
      }
      // size_buffer will be filled.
      int64_t gap = sizeof(int64_t) - buf->size_offset;
      memcpy(
        buf->size_buffer + buf->size_offset,
        data, gap);
      // size_offset == sizeof(int64_t)
      buf->size_offset += gap;
      // length and data will be reused for filling data_buffer.
      length = length - gap;
      data = data + gap;
      // We got data_size from size_buffer
      memcpy(&buf->data_size, buf->size_buffer, sizeof(int64_t));
      // Allocate data_buffer
      buf->data_buffer = reinterpret_cast<char *>(malloc(buf->data_size));
    }
    // If data_buffer will be not filled in this iter, finish.
    if (buf->data_offset + length < buf->data_size) {
      memcpy(
        buf->data_buffer + buf->data_offset,
        data, length);
      buf->data_offset += length;
      return msgs;
    }
    // data_buffer will be filled.
    int64_t gap = buf->data_size - buf->data_offset;
    memcpy(
      buf->data_buffer + buf->data_offset,
      data, gap);
    // data_offset == data_size
    buf->data_offset += gap;
    // length and data will be reused for filling size_buffer.
    length = length - gap;
    data = data + gap;
    msgs.emplace_back(buf->data_buffer, buf->data_size);
    // Reset UCXStreamBuffer
    buf->data_size = 0;
    buf->size_offset = 0;
    buf->data_buffer = NULL;
    buf->data_offset = 0;
  }
  return msgs;
}

void UCXReceiver::RecvLoop(UCXReceiver *self) {
  std::vector<UCXStreamBuffer> streambuf(
    self->eps_.size(), UCXStreamBuffer());
  std::vector<CircularBufferProducer<Message>> producers;
  for (int id=0; id < self->eps_.size(); id++) {
    producers.emplace_back(self->recv_queues_[id]);
  }
  CircularBufferProducer<int> notify(self->notify_queue_);
  size_t length;
  ucs_status_ptr_t stat;
  while (true) {
    for (int id=0; id < self->eps_.size(); id++) {
      stat = ucp_stream_recv_data_nb(self->eps_[id], &length);
      if (stat == NULL) {
        continue;
      }
      if (UCS_PTR_IS_ERR(stat)) {
        LOG(FATAL) << "ucp_stream_recv_data_nb failed: sender_id=" << id
        << ucs_status_string(UCS_PTR_STATUS(stat));
        continue;
      }
      std::vector<Message> msgs = RecvMsgLoop(&streambuf[id], length, stat);
      ucp_stream_data_release(self->eps_[id], stat);
      if (!msgs.empty()) {
        for (auto &msg : msgs) {
          producers[id].push(std::move(msg));
        }
        notify.push(id);
      }
    }
  }
}

STATUS UCXReceiver::Recv(Message* msg, int* send_id) {
  while (true) {
    while (notify_consumer_->empty()) {
    }
    int id = notify_consumer_->pop();
    if (recv_consumer_[id]->empty()) {
      continue;
    }
    *send_id = id;
    break;
  }
  LOG(INFO) << "id=" << *send_id << " message received";
  *msg = recv_consumer_[*send_id]->pop();
  return REMOVE_SUCCESS;
}

STATUS UCXReceiver::RecvFrom(Message* msg, int send_id) {
  while (recv_consumer_[send_id]->empty()) {
  }
  *msg = recv_consumer_[send_id]->pop();
  return REMOVE_SUCCESS;
}

}  // namespace network
}  // namespace dgl
