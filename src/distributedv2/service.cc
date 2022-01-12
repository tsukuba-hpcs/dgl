#include "service.h"

#include <memory>
#include <thread>

namespace dgl {
namespace distributedv2 {

ServiceManager::ServiceManager(int rank, int size, Communicator *comm)
  : shutdown(false)
  , rank(rank)
  , size(size)
  , comm(comm)
  , progress_counter(0) {
  servs.reserve(MAX_SERVICE_LEN);
  args.reserve(MAX_SERVICE_LEN);
#ifdef DGL_USE_NVTX
  nvtxNameOsThread(syscall(SYS_gettid), "Main Thread");
#endif // DGL_USE_NVTX
}

void ServiceManager::am_recv_cb(void *arg, const void *buffer, size_t length) {
  sm_cb_arg_t *cbarg = (sm_cb_arg_t *)arg;
  ((AMService *)(cbarg->serv))->am_recv(cbarg->comm, buffer, length);
}

void ServiceManager::add_am_service(std::unique_ptr<AMService> &&serv) {
  servs.push_back(std::move(serv));
  sm_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg));
  unsigned am_id = comm->add_am_handler(&args.back(), am_recv_cb);
  CHECK(am_id < MAX_SERVICE_LEN);
 ((AMService *)servs.back().get())->am_id = am_id;
}

void ServiceManager::rma_recv_cb(void *arg, uint64_t req_id) {
  sm_cb_arg_t *cbarg = (sm_cb_arg_t *)arg;
  ((RMAService *)cbarg->serv)->rma_read_cb(cbarg->comm, req_id);
}

void ServiceManager::add_rma_service(std::unique_ptr<RMAService> &&serv) {
  servs.push_back(std::move(serv));
  sm_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg));
  auto buf = ((RMAService *)servs.back().get())->served_buffer();
  unsigned rma_id = comm->add_rma_handler(buf.first, buf.second, &args.back(), rma_recv_cb);
  CHECK(rma_id < MAX_SERVICE_LEN);
  ((RMAService *)servs.back().get())->rma_id = rma_id;
}

rma_mem_ret_t ServiceManager::map_rma_service() {
  return comm->rma_mem_map();
}

void ServiceManager::prepare_rma_service(void *rkeybuf, size_t rkeybuf_len, void *address, size_t address_len) {
  comm->prepare_rma(rkeybuf, rkeybuf_len, address, address_len);
}

void ServiceManager::add_stub_service(std::unique_ptr<StubService> &&serv) {
  serv->stub_id = servs.size();
  servs.push_back(std::move(serv));
  sm_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg));
}

void ServiceManager::progress() {
  for (auto &serv: servs) {
    act_counter += serv->progress(comm);
  }
  act_counter += comm->progress();
  progress_counter++;
  if (progress_counter % COUNTER_THRESHOLD == 0) {
    LOG(INFO) << "rank=" << rank << " ServiceManager::progress()";
  }
  if (progress_counter % YIELD_THRESHOLD == 0) {
    if (act_counter == 0) {
      std::this_thread::yield();
    }
    act_counter = 0;
  }
}

void ServiceManager::run(ServiceManager *self) {
#ifdef DGL_USE_NVTX
  nvtxNameOsThread(syscall(SYS_gettid), "UCX Thread");
#endif // DGL_USE_NVTX
  while (!self->shutdown) {
    self->progress();
  }
}

void ServiceManager::launch() {
  progress_thread = std::thread(run, this);
}

void ServiceManager::terminate() {
  shutdown.store(true);
  progress_thread.join();
}

}
}