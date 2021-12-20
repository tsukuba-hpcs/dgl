#include "comm.h"

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"

namespace dgl {
namespace distributedv2 {

using namespace dgl::runtime;

Communicator::Communicator(int rank, int size, size_t buffer_len)
: state(CommState::INIT)
, rank(rank)
, size(size)
, am_pool(buffer_len)
, rma_pool(buffer_len) {
  ucs_status_t status;
  ucp_params_t params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS,
    .features = UCP_FEATURE_AM | UCP_FEATURE_RMA,
    .estimated_num_eps = static_cast<size_t>(size),
  };
  if ((status = ucp_init(&params, NULL, &ucp_context)) != UCS_OK) {
    LOG(FATAL) << "ucp_init error: " << ucs_status_string(status);
  }
}

std::pair<ucp_address_t*, int> Communicator::create_workers() {
  CHECK(state == CommState::INIT);
  ucs_status_t status;
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
  ucp_worker_print_info(ucp_worker, stdout);
  state = CommState::WORKER_READY;

  for (unsigned am_id = 0; am_id < am_handlers.size(); am_id++) {
    ucp_am_handler_param_t param = {
      .field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_ARG | UCP_AM_HANDLER_PARAM_FIELD_CB,
      .id = am_id,
      .cb = recv_cb,
      .arg = &am_handlers[am_id],
    };
    status = ucp_worker_set_am_recv_handler(ucp_worker, &param);
    if (status != UCS_OK) {
      LOG(FATAL) << "ucp_worker_set_am_recv_handler failed with "
        << ucs_status_string(status);
    }
  }
  chunks.assign(size, std::vector<iov_pool_item_t *>(am_handlers.size(), NULL));

  return std::make_pair(addr, static_cast<int>(addrlen));
}

void Communicator::create_endpoints(void *addrs, size_t length) {
  CHECK(state == CommState::WORKER_READY);
  CHECK(length == size * addrlen);
  ucs_status_t status;
  eps.resize(size);
  for (int cur = 0; cur != size; cur++) {
    if (cur == rank) continue;
    const ucp_address_t* addr =
      reinterpret_cast<const ucp_address_t *>(UCS_PTR_BYTE_OFFSET(addrs, addrlen * cur));
    ucp_ep_params_t params = {
      .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE,
      .address = addr,
      .err_mode = UCP_ERR_HANDLING_MODE_PEER,
    };
    if ((status = ucp_ep_create(ucp_worker, &params, &eps[cur])) != UCS_OK) {
      LOG(FATAL) << "rank=" << rank
        <<"ucp_worker_get_address error: " << ucs_status_string(status);
    }
    ucp_ep_print_info(eps[cur], stdout);
  }
  state = CommState::EP_READY;
  if (rma_handlers.empty()) {
    state = CommState::READY;
  }
}

Communicator::~Communicator() {
  CHECK(state == CommState::READY);
  ucs_status_t status;
  for (int rma_id = 0; rma_id < rma_handlers.size(); rma_id++) {
    ucp_rkey_buffer_release(rma_handlers[rma_id].rkeybuf);
    ucp_mem_attr_t attr = {
      .field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH,
    };
    status = ucp_mem_query(rma_handlers[rma_id].mem, &attr);
    if (status != UCS_OK) {
      LOG(FATAL) << "ucp_mem_query failed with " << ucs_status_string(status);
    }
    for (int srcrank = 0; srcrank < size; srcrank++) {
      if (srcrank == rank) continue;
      ucp_rkey_destroy(rma_handlers[rma_id].rkeys[srcrank]);
    }
    status = ucp_mem_unmap(ucp_context, rma_handlers[rma_id].mem);
    if (status != UCS_OK) {
      LOG(FATAL) << "ucp_mem_unmap failed with " << ucs_status_string(status);
    }
  }
  ucs_status_ptr_t req;
  ucp_request_param_t params = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
    .flags = UCP_EP_CLOSE_MODE_FLUSH,
  };
  for (int cur = 0; cur < size; cur++) {
    if (cur == rank) continue;
    req = ucp_ep_close_nbx(eps[cur], &params);
    if (req == NULL) continue;
    if (req != NULL && UCS_PTR_IS_ERR(req)) {
      LOG(FATAL) << "ucp_ep_close_nbx failed with"
        << ucs_status_string(UCS_PTR_STATUS(req));
    }
    status = ucp_request_check_status(req);
    if (status == UCS_INPROGRESS) {
      do {
        ucp_worker_progress(ucp_worker);
        status = UCS_PTR_STATUS(req);
      } while (status == UCS_INPROGRESS && req != NULL);
      if (status != UCS_OK) {
        LOG(FATAL) << "ucp_ep_close_nbx failed with"
          << ucs_status_string(status);
      }
    }
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
  if (p->used) {
    LOG(INFO) << "IovPool alloc item failed: cursor= " << cursor;
    return 1;
  }
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
  am_handler_t *handler = (am_handler_t *)(arg);
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

void Communicator::am_send(int rank, unsigned am_id, iov_pool_item_t *chunk) {
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
  status = ucp_am_send_nbx(eps[rank], am_id,
    chunk->header, header_length, chunk->iov, chunk->iov_cnt, &params);
  if (status == NULL) {
    chunk->release();
    return;
  }
  if (UCS_PTR_IS_ERR(status)) {
    chunk->release();
    LOG(FATAL) << "ucp_am_send_nbx failed with " << ucs_status_string(UCS_PTR_STATUS(status));
  }
}

void Communicator::am_post(int destrank, unsigned am_id, std::unique_ptr<uint8_t[]> &&data, size_t length) {
  CHECK(state == CommState::READY);
  if (chunks[destrank][am_id] == NULL) {
    CHECK(!am_pool.alloc(&chunks[destrank][am_id]));
  }
  chunks[destrank][am_id]->append(std::move(data), length);
  if (chunks[destrank][am_id]->filled()) {
    am_send(destrank, am_id, chunks[destrank][am_id]);
    chunks[destrank][am_id] = NULL;
  }
}

unsigned Communicator::add_am_handler(void *arg, comm_am_cb_t cb) {
  CHECK(state == CommState::INIT);
  ucs_status_t status;
  unsigned am_id = am_handlers.size();
  am_handlers.push_back(am_handler_t{.arg = arg, .cb = cb});
  return am_id;
}

void Communicator::progress() {
  CHECK(state == CommState::READY);
  for (int destrank = 0; destrank < size; destrank++) {
    if (destrank == rank) continue;
    for (unsigned am_id = 0; am_id < am_handlers.size(); am_id++) {
      if (chunks[destrank][am_id] == NULL) continue;
      if (chunks[destrank][am_id]->empty()) continue;
      am_send(destrank, am_id, chunks[destrank][am_id]);
      chunks[destrank][am_id] = NULL;
    }
  }
  ucp_worker_progress(ucp_worker);
}

rma_pool_item_t::rma_pool_item_t(rma_handler_t *handler = NULL)
: used(false)
, handler(handler) {
}

void rma_pool_item_t::release() {
  CHECK(used);
  used = false;
}

RmaPool::RmaPool(size_t length)
: buffer(length), cursor(0) {
}

int RmaPool::alloc(rma_pool_item_t** item, size_t req_id, void *address, rma_handler_t *handler) {
  rma_pool_item_t *p = &buffer[cursor];
  if (p->used) {
    LOG(INFO) << "RmaPool alloc item failed: cursor= " << cursor;
    int used = 0;
    for (size_t c = 0; c < buffer.size(); c++) {
      if (buffer[c].used) used++;
    }
    LOG(INFO) << "RmaPool used=" << used << " unused=" << buffer.size() - used;
    return 1;
  }
  p->used = true;
  p->req_id = req_id;
  p->address = address;
  p->handler = handler;
  *item = p;
  cursor = (cursor + 1) % buffer.size();
  return 0;
}

unsigned Communicator::add_rma_handler(void *buffer, size_t length, void *arg, comm_rma_cb_t cb) {
  CHECK(state == CommState::INIT);
  unsigned rma_id = rma_handlers.size();
  rma_handlers.push_back(rma_handler_t(arg, cb, buffer, length));
  rma_handlers.back().rkeys.resize(size);
  rma_handlers.back().addresses.resize(size);
  return rma_id;
}

rma_mem_ret_t Communicator::rma_mem_map() {
  ucs_status_t status;
  CHECK(state == CommState::EP_READY);
  CHECK(rma_rkeybuf.empty());
  CHECK(rma_address.empty());
  int rkeybuf_len = 0;
  int address_len = 0;
  for (unsigned rma_id = 0; rma_id < rma_handlers.size(); rma_id++) {
    ucp_mem_map_params_t params = {
      .field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH,
      .address = rma_handlers[rma_id].address,
      .length = rma_handlers[rma_id].buffer_len,
    };
    status = ucp_mem_map(ucp_context, &params, &rma_handlers[rma_id].mem);
    if (status != UCS_OK) {
      LOG(FATAL) << "ucp_mem_map failed with "
        << ucs_status_string(status);
    }
    status = ucp_rkey_pack(ucp_context, rma_handlers[rma_id].mem,
      &rma_handlers[rma_id].rkeybuf, &rma_handlers[rma_id].rkeybuf_len);
    if (status != UCS_OK) {
      LOG(FATAL) << "ucp_rkey_pack failed with "
        << ucs_status_string(status);
    }
    rkeybuf_len += rma_handlers[rma_id].rkeybuf_len;
    address_len += sizeof(rma_handlers[rma_id].address);
  }
  rma_rkeybuf.resize(rkeybuf_len);
  rma_address.resize(address_len);
  int rkeybuf_offset = 0;
  int address_offset = 0;
  for (unsigned rma_id = 0; rma_id < rma_handlers.size(); rma_id++) {
    std::memcpy(&rma_rkeybuf[rkeybuf_offset], rma_handlers[rma_id].rkeybuf, rma_handlers[rma_id].rkeybuf_len);
    rkeybuf_offset += rma_handlers[rma_id].rkeybuf_len;
    std::memcpy(&rma_address[address_offset], &rma_handlers[rma_id].address, sizeof(rma_handlers[rma_id].address));
    address_offset += sizeof(rma_handlers[rma_id].address);
  }
  CHECK(rkeybuf_offset == rkeybuf_len);
  CHECK(address_offset == address_len);
  state = CommState::MEM_MAPPED;
  return rma_mem_ret_t{
    .rkeybuf = &rma_rkeybuf[0],
    .rkeybuf_len = rkeybuf_len,
    .address = &rma_address[0],
    .address_len = address_len
  };
}

void Communicator::read_cb(void *request, ucs_status_t status, void *user_data) {
  if (status != UCS_OK) {
    LOG(FATAL) << "read_cb failed with " << ucs_status_string(status);
    return;
  }
  rma_pool_item_t *item = (rma_pool_item_t *)user_data;
  item->handler->cb(item->handler->arg, item->req_id, item->address);
  item->release();
  ucp_request_free(request);
}

void Communicator::rma_read(int destrank, unsigned rma_id, uint64_t req_id, void *buffer, uint64_t offset, size_t length) {
  CHECK(state == CommState::READY);
  ucs_status_ptr_t status;
  rma_pool_item_t *item;
  rma_handler_t *handler = &rma_handlers[rma_id];
  CHECK(!rma_pool.alloc(&item, req_id, buffer, handler));
  ucp_request_param_t params = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA,
    .cb = {
      .send = read_cb,
    },
    .user_data = item,
  };
  status = ucp_get_nbx(eps[destrank], buffer, length, handler->addresses[destrank] + offset, handler->rkeys[destrank], &params);
  if (status == NULL) {
    handler->cb(handler->arg, req_id, buffer);
    item->release();
    return;
  }
  if (UCS_PTR_IS_ERR(status)) {
    item->release();
    LOG(FATAL) << "ucp_get_nbx failed with " << ucs_status_string(UCS_PTR_STATUS(status));
    return;
  }
  return;
}

void Communicator::prepare_rma(void *rkeybuf, size_t rkeybuf_len, void *address, size_t address_len) {
  CHECK(rma_rkeybuf.size() * size == rkeybuf_len);
  CHECK(rma_address.size() * size == address_len);
  CHECK(state == CommState::MEM_MAPPED);
  ucs_status_t status;
  size_t rkeybuf_offset = 0;
  size_t address_offset = 0;
  for (unsigned rma_id = 0; rma_id < rma_handlers.size(); rma_id++) {
    for (int srcrank = 0; srcrank < size; srcrank++) {
      if (srcrank == rank) {
        continue;
      }
      status = ucp_ep_rkey_unpack(
        eps[srcrank],
        UCS_PTR_BYTE_OFFSET(rkeybuf, rma_address.size() * srcrank + rkeybuf_offset),
        &rma_handlers[rma_id].rkeys[srcrank]);
      if (status != UCS_OK) {
        LOG(FATAL) << "ucp_ep_rkey_unpack failed with "
        << ucs_status_string(status);
      }
      std::memcpy(
        &rma_handlers[rma_id].addresses[srcrank],
        UCS_PTR_BYTE_OFFSET(address, rma_address.size() * srcrank + address_offset),
        sizeof(rma_handlers[rma_id].address)
      );
    }
    rkeybuf_offset += rma_handlers[rma_id].rkeybuf_len;
    address_offset += sizeof(rma_handlers[rma_id].address);
  }
  state = CommState::READY;
}



DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateCommunicator")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "_CAPI_DistV2CreateCommunicator is called";
  int rank = args[0];
  int size = args[1];
  std::shared_ptr<Communicator> ctx(new Communicator(rank, size));
  *rv = ctx;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateWorker")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  CommunicatorRef comm = args[0];
  List<Value> ret;
  auto p = comm->create_workers();
  ret.push_back(Value(MakeValue(p.first)));
  ret.push_back(Value(MakeValue(p.second)));
  *rv = ret;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateEndpoints")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  CommunicatorRef comm = args[0];
  std::string addrs = args[1];
  comm->create_endpoints(&addrs[0], addrs.size());
});


}
}