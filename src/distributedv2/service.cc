#include "service.h"
#include "context.h"

#include <sstream>
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
  serv->am_id = servs.size();
  servs.push_back(std::move(serv));
  sm_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg)); 
  unsigned id = comm->add_am_handler(&args.back(), am_recv_cb);
  CHECK(id == ((AMService *)servs.back().get())->am_id);
}

void ServiceManager::add_rma_service(std::unique_ptr<RMAService> &&serv) {
  
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