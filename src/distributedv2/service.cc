#include "service.h"
#include "context.h"

#include <sstream>
#include <memory>

namespace dgl {
namespace distributedv2 {


GraphServer::GraphServer(GraphRef g)
  : local_graph(g) {
}

void GraphServer::recv(Communicator *comm, const void *buffer, size_t length) {

}

void GraphServer::progress(Communicator *comm) {

}

ServiceManager::ServiceManager(int rank, int size, Communicator *comm)
  : shutdown(false)
  , rank(rank)
  , size(size)
  , comm(comm) {
}

void ServiceManager::recv_cb(void *arg, comm_iov_t *iov, uint8_t iov_cnt) {
  sm_recv_cb_arg_t *cbarg = (sm_recv_cb_arg_t *)arg;
  for (uint8_t idx = 0; idx < iov_cnt; idx++) {
    cbarg->serv->recv(cbarg->comm, iov[idx].buffer, iov[idx].length);
  }
}

void ServiceManager::add_service(std::unique_ptr<Service> &&serv) {
  servs.push_back(std::move(serv));
  sm_recv_cb_arg_t arg(servs.back().get(), comm);
  args.push_back(std::move(arg));
  unsigned id = comm->add_recv_handler(&args.back(), recv_cb);
  CHECK(id + 1 == servs.size());
}

void ServiceManager::run(ServiceManager *self) {
  size_t length;
  while (!self->shutdown) {
    for (auto &serv: self->servs) {
      serv->progress(self->comm);
    }
    self->comm->progress();
  }
}

}
}