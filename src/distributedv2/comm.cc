#include "comm.h"

namespace dgl {
namespace distributedv2 {


Communicator::Communicator(int rank, int size)
: rank(rank)
, size(size) {
  ucs_status_t status;
  ucp_params_t params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS,
    .features = UCP_FEATURE_STREAM | UCP_FEATURE_RMA,
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

void Communicator::get_data(int srcrank, std::stringstream *ss) {
  ucs_status_ptr_t stat;
  size_t length;
  stat = ucp_stream_recv_data_nb(eps[srcrank], &length);
  if (length == 0 || stat == NULL) return;
  ss->write(reinterpret_cast<const char*>(stat), length);
  ucp_stream_data_release(eps[srcrank], stat);
}

void Communicator::progress() {
  ucp_worker_progress(ucp_worker);
}



}
}