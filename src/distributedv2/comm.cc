#include "comm.h"

namespace dgl {
namespace distributedv2 {


Communicator::Communicator(int rank, int size, size_t buffer_len)
: rank(rank)
, size(size)
, pool(buffer_len) {
  ucs_status_t status;
  ucp_params_t params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS,
    .features = UCP_FEATURE_AM | UCP_FEATURE_RMA,
    .estimated_num_eps = static_cast<size_t>(size),
  };
  if ((status = ucp_init(&params, NULL, &ucp_context)) != UCS_OK) {
    LOG(FATAL) << "ucp_init error: " << ucs_status_string(status);
  }
  ucp_worker_params_t wparams = {
    .field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
    .thread_mode = UCS_THREAD_MODE_SERIALIZED,
  };
  if ((status = ucp_worker_create(ucp_context, &wparams, &ucp_worker)) != UCS_OK) {
    LOG(FATAL) << "ucp_worker_create error: " << ucs_status_string(status);
  }
  if ((status = ucp_worker_get_address(ucp_worker, &addr, &addrlen)) != UCS_OK) {
    LOG(FATAL) << "ucp_worker_get_address error: " << ucs_status_string(status);
  }
}

ucp_address_t* Communicator::get_workeraddr() {
  return addr;
}

int Communicator::get_workerlen() {
  return static_cast<int>(addrlen);
}

void Communicator::create_endpoints(std::string addrs) {
  CHECK(addrs.length() == size * addrlen);
  ucs_status_t status;
  eps.resize(size);
  for (int cur = 0; cur != size; cur++) {
    if (cur == rank) continue;
    const ucp_address_t* addr =
      reinterpret_cast<const ucp_address_t *>(&addrs[addrlen * cur]);
    ucp_ep_params_t params = {
      .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS,
      .address = addr,
    };
    if ((status = ucp_ep_create(ucp_worker, &params, &eps[cur])) != UCS_OK) {
      LOG(FATAL) << "rank=" << rank
        <<"ucp_worker_get_address error: " << ucs_status_string(status);
    }
  }
}

Communicator::~Communicator() {
  for (int cur = 0; cur != size; cur++) {
    if (cur == rank) continue;
    ucp_ep_destroy(eps[cur]);
  }
  ucp_worker_release_address(ucp_worker, addr);
  ucp_worker_destroy(ucp_worker);
  ucp_cleanup(ucp_context);
}

iov_pool_item_t::iov_pool_item_t()
: used(false)
, iov_cnt(0) {
  std::fill(iov, iov + MAX_IOV_CNT, ucp_dt_iov_t{.buffer = NULL, .length = 0});
  std::fill(data, data + MAX_IOV_CNT, nullptr);
}

void iov_pool_item_t::release() {
  CHECK(used);
  used = false;
  for (uint8_t idx = 0; idx < iov_cnt; idx++) {
    data[idx] = nullptr;
  }
}

size_t iov_pool_item_t::fill_header() {
  std::fill(header, header + HEADER_LEN, (uint8_t)0);
  size_t offset = 0;
  *((uint8_t *)UCS_PTR_BYTE_OFFSET(header, offset)) = iov_cnt;
  offset += sizeof(uint8_t);
  for (uint8_t idx = 0; idx < iov_cnt; idx++) {
    *((size_t *)UCS_PTR_BYTE_OFFSET(header, offset)) = iov[idx].length;
    offset += sizeof(size_t);
  }
  return offset;
}

bool iov_pool_item_t::empty() {
  return iov_cnt == 0;
}

bool iov_pool_item_t::filled() {
  return iov_cnt == MAX_IOV_CNT;
}

void iov_pool_item_t::append(std::unique_ptr<uint8_t[]> &&buf, size_t length) {
  CHECK(iov_cnt < MAX_IOV_CNT);
  iov[iov_cnt].buffer = buf.get();
  iov[iov_cnt].length = length;
  data[iov_cnt] = std::move(buf);
  iov_cnt++;
}

IovPool::IovPool(size_t length)
: buffer(length), cursor(0) {}

int IovPool::alloc(iov_pool_item_t** item) {
  iov_pool_item_t *p = &buffer[cursor];
  if (p->used) return 1;
  p->used = true;
  std::memset(p->iov, 0, sizeof(p->iov));
  p->iov_cnt = 0;
  *item = p;
  cursor = (cursor + 1) % buffer.size();
  return 0;
}



ucs_status_t Communicator::recv_cb(
  void *arg,
  const void *header,
  size_t header_length,
  void *data, size_t length,
  const ucp_am_recv_param_t *param) {
  CHECK(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
  CHECK(header_length >= sizeof(uint8_t));
  uint8_t iov_cnt = *((uint8_t*)header);
  CHECK(header_length == sizeof(uint8_t) + sizeof(size_t) * iov_cnt);
  size_t *iov_len = (size_t *)UCS_PTR_BYTE_OFFSET(header, sizeof(uint8_t));
  comm_recv_handler_t *handler = (comm_recv_handler_t *)(arg);
  size_t offset = 0;

  for (size_t idx = 0; idx < iov_cnt; idx++) {
    (*handler->cb)(handler->arg, UCS_PTR_BYTE_OFFSET(data, offset), iov_len[idx]);
    offset += iov_len[idx];
  }
  CHECK(offset == length);
  return UCS_OK;
}

void Communicator::send_cb(void *request, ucs_status_t status, void *user_data) {
  if (status != UCS_OK) {
    LOG(FATAL) << "send_cb failed with " << ucs_status_string(status);
  }
  iov_pool_item_t *chunk = (iov_pool_item_t *)user_data;
  chunk->release();
  ucp_request_free(request);
}

void Communicator::send(int rank, unsigned id, iov_pool_item_t *chunk) {
  ucp_request_param_t params = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_FLAGS | UCP_OP_ATTR_FIELD_USER_DATA,
    .flags = UCP_AM_SEND_FLAG_EAGER,
    .cb = {
      .send = send_cb,
    },
    .datatype = UCP_DATATYPE_IOV,
    .user_data = chunk,
  };
  ucs_status_ptr_t status;
  size_t header_length;
  header_length = chunk->fill_header();
  status = ucp_am_send_nbx(eps[rank], id,
    chunk->header, header_length, chunk->iov, chunk->iov_cnt, &params);
  if (status == NULL) return;
  if (UCS_PTR_IS_ERR(status)) {
    LOG(FATAL) << "ucp_am_send_nbx failed with " << ucs_status_string(UCS_PTR_STATUS(status));
  }
}

void Communicator::append(int destrank, unsigned id, std::unique_ptr<uint8_t[]> &&data, size_t length) {
  if (chunks[destrank][id] == NULL) {
    CHECK(!pool.alloc(&chunks[destrank][id]));
  }
  if (chunks[destrank][id]->filled()) {
    send(destrank, id, chunks[destrank][id]);
    CHECK(!pool.alloc(&chunks[destrank][id]));
  }
  chunks[destrank][id]->append(std::move(data), length);
}

unsigned Communicator::add_recv_handler(void *arg, comm_cb_handler_t cb) {
  ucs_status_t status;
  unsigned id = recv_handlers.size();
  recv_handlers.push_back({arg, cb});
  ucp_am_handler_param_t param = {
    .field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_ARG | UCP_AM_HANDLER_PARAM_FIELD_CB,
    .id = id,
    .cb = recv_cb,
    .arg = &recv_handlers.back(),
  };
  status = ucp_worker_set_am_recv_handler(ucp_worker, &param);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_worker_set_am_recv_handler failed with "
      << ucs_status_string(status);
  }
  chunks.assign(size, std::vector<iov_pool_item_t *>(recv_handlers.size(), NULL));
  return id;
}

void Communicator::progress() {
  ucp_worker_progress(ucp_worker);
  for (int destrank = 0; destrank < size; destrank++) {
    if (destrank == rank) continue;
    for (unsigned id = 0; id < recv_handlers.size(); id++) {
      if (chunks[destrank][id] == NULL) continue;
      if (chunks[destrank][id]->empty()) continue;
      send(destrank, id, chunks[destrank][id]);
      chunks[destrank][id] = NULL;
    }
  }
}



}
}