#include "service.h"
#include "context.h"

#include <memory>

namespace dgl {
namespace distributedv2 {

ServiceManager::ServiceManager(int rank, int size, Communicator *comm)
  : shutdown(false)
  , rank(rank)
  , size(size)
  , comm(comm) {
}

void ServiceManager::am_recv_cb(void *arg, const void *buffer, size_t length) {
  sm_cb_arg_t *cbarg = (sm_cb_arg_t *)arg;
  ((AMService *)cbarg->serv)->am_recv(cbarg->comm, buffer, length);
}

void ServiceManager::add_am_service(std::unique_ptr<AMService> &&serv) {
  servs.push_back(std::move(serv));
  sm_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg)); 
  unsigned am_id = comm->add_am_handler(&args.back(), am_recv_cb);
 ((AMService *)servs.back().get())->am_id = am_id;
}

void ServiceManager::rma_recv_cb(void *arg, uint64_t req_id, void *address) {
  sm_cb_arg_t *cbarg = (sm_cb_arg_t *)arg;
  ((RMAService *)cbarg->serv)->rma_read_cb(cbarg->comm, req_id, address);
}

rma_serv_ret_t ServiceManager::add_rma_service(std::unique_ptr<RMAService> &&serv) {
  servs.push_back(std::move(serv));
  sm_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg));
  auto buf = ((RMAService *)servs.back().get())->served_buffer();
  unsigned rma_id = comm->add_rma_handler(buf.first, buf.second, &args.back(), rma_recv_cb);
  ((RMAService *)servs.back().get())->rma_id = rma_id;
  auto r = comm->get_rkey_buf(rma_id);
  return rma_serv_ret_t{.rma_id = rma_id, .rkey_buf = r.first, .rkey_buf_len = r.second};
}

void ServiceManager::setup_rma_service(unsigned rma_id, void *rkey_bufs, size_t rkey_buf_len, void *address, size_t addr_len) {
  comm->create_rkey(rma_id, rkey_bufs, rkey_buf_len);
  comm->set_buffer_addr(rma_id, (intptr_t)address, addr_len);
}

void ServiceManager::add_stub_service(std::unique_ptr<StubService> &&serv) {
  serv->stub_id = servs.size();
  servs.push_back(std::move(serv));
  sm_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg));
}

void ServiceManager::progress() {
  for (auto &serv: servs) {
    serv->progress(comm);
  }
  comm->progress();
}

void ServiceManager::run(ServiceManager *self) {
  while (!self->shutdown) {
    self->progress();
  }
}

}
}