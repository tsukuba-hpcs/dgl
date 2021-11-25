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

IovPool::IovPool(size_t length)
: buffer(length), cursor(0) {}

int IovPool::alloc(iov_pool_item_t** item) {
  iov_pool_item_t *p = &buffer[cursor];
  if (p->used) return 1;
  p->used = true;
  for (uint8_t idx = 0; idx < p->iov_cnt; idx++) {
    if (p->data[idx] == NULL) continue;
    free(p->data[idx]);
    p->data[idx] = NULL;
  }
  std::memset(p->iov, 0, sizeof(p->iov));
  p->iov_cnt = 0;
  *item = p;
  cursor = (cursor + 1) % buffer.size();
  return 0;
}

void IovPool::release(iov_pool_item_t* item) {
  CHECK(item->used);
  item->used = false;
}

size_t IovPool::fill_header(iov_pool_item_t* item) {
  std::memset(item->header, 0, sizeof(item->header));
  size_t offset = 0;
  *((uint8_t *)UCS_PTR_BYTE_OFFSET(item->header, offset)) = item->iov_cnt;
  offset += sizeof(uint8_t);
  for (uint8_t idx = 0; idx < item->iov_cnt; idx++) {
    *((size_t *)UCS_PTR_BYTE_OFFSET(item->header, offset)) = item->iov[idx].length;
    offset += sizeof(size_t);
  }
  return offset;
}

void IovPool::append(iov_pool_item_t* item, void *data, size_t length) {
  CHECK(item->iov_cnt < MAX_IOV_CNT);
  item->data[item->iov_cnt] = data;
  item->iov[item->iov_cnt].buffer = data;
  item->iov[item->iov_cnt].length = length;
  item->iov_cnt++;
}

bool IovPool::filled(iov_pool_item_t* item) {
  return item->iov_cnt == MAX_IOV_CNT;
}

bool IovPool::empty(iov_pool_item_t* item) {
  if (item == NULL) return true;
  return item->iov_cnt == 0;
}


ucs_status_t Communicator::recv_cb(
  void *arg,
  const void *header,
  size_t header_length,
  void *data, size_t length,
  const ucp_am_recv_param_t *param) {
  CHECK(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
  CHECK(header_length >= sizeof(uint8_t));
  iov_pool_item_t *item;
  size_t idx;
  size_t offset;
  uint8_t iov_cnt = ((uint8_t*)header)[0];
  CHECK(header_length == sizeof(uint8_t) + sizeof(size_t) * iov_cnt);
  size_t *iov_len = (size_t *)UCS_PTR_BYTE_OFFSET(header, sizeof(uint8_t));
  comm_recv_handler_t *handler = (comm_recv_handler_t *)(arg);

  CHECK(!handler->pool->alloc(&item));
  item->iov_cnt = iov_cnt;
  offset = 0;

  for (idx = 0; idx < item->iov_cnt; idx++) {
    item->iov[idx].length = iov_len[idx];
    item->iov[idx].buffer = UCS_PTR_BYTE_OFFSET(data, offset);
    offset += item->iov[idx].length;
  }
  CHECK(offset == length);

  (*handler->cb)(handler->arg, item->iov, item->iov_cnt);

  handler->pool->release(item);
  return UCS_OK;
}

void Communicator::send_cb(void *request, ucs_status_t status, void *user_data) {
  if (status != UCS_OK) {
    LOG(FATAL) << "send_cb failed with " << ucs_status_string(status);
  }
  comm_chunk_t *chunk = (comm_chunk_t *)user_data;
  chunk->pool->release(chunk->item);
  ucp_request_free(request);
}

void Communicator::send(int rank, unsigned id, comm_chunk_t *chunk) {
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
  header_length = chunk->pool->fill_header(chunk->item);
  status = ucp_am_send_nbx(eps[rank], id,
    chunk->item->header, header_length, chunk->item->iov, chunk->item->iov_cnt, &params);
  // realloc chunk
  CHECK(!pool.alloc(&chunk->item));
  if (status == NULL) return;
  if (UCS_PTR_IS_ERR(status)) {
    LOG(FATAL) << "ucp_am_send_nbx failed with " << ucs_status_string(UCS_PTR_STATUS(status));
  }
}

void Communicator::append(int rank, unsigned id, void *data, size_t length) {
  comm_chunk_t *chunk = &chunks[rank][id];
  if (chunk->item == NULL) {
    CHECK(!pool.alloc(&chunk->item));
  }
  if (pool.filled(chunk->item)) {
    send(rank, id, chunk);
  }
  pool.append(chunk->item, data, length);
}

unsigned Communicator::add_recv_handler(void *arg, comm_cb_handler_t cb) {
  ucs_status_t status;
  unsigned id = recv_handlers.size();
  recv_handlers.push_back({arg, cb, &pool});
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
  chunks.assign(size, std::vector<comm_chunk_t>(recv_handlers.size(), { NULL, &pool}));
  return id;
}

void Communicator::progress() {
  ucp_worker_progress(ucp_worker);
  for (int destrank = 0; destrank < size; destrank++) {
    if (destrank == rank) continue;
    for (unsigned id = 0; id < recv_handlers.size(); id++) {
      if (pool.empty(chunks[destrank][id].item)) continue;
      send(destrank, id, &chunks[destrank][id]);
    }
  }
}



}
}