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

void ServiceManager::recv_cb(void *arg, const void *buffer, size_t length) {
  sm_recv_cb_arg_t *cbarg = (sm_recv_cb_arg_t *)arg;
  cbarg->serv->recv(cbarg->comm, buffer, length);
}

void ServiceManager::add_service(std::unique_ptr<Service> &&serv) {
  serv->sid = servs.size();
  servs.push_back(std::move(serv));
  sm_recv_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg));
  unsigned id = comm->add_recv_handler(&args.back(), recv_cb);
  CHECK(id == servs.back()->sid);
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