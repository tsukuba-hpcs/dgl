#include "service.h"
#include "context.h"

#include <ucp/api/ucp.h>
#include <sstream>

namespace dgl {
namespace distributedv2 {


GraphServer::GraphServer(GraphRef g)
  : local_graph(g) {
}

void GraphServer::recv(Communicator *comm, EndpointState *estate) {

}

void GraphServer::progress(Communicator *comm) {

}

ServiceManager::ServiceManager(int rank, int size)
  : shutdown(false)
  , rank(rank)
  , size(size) {
  for(int rank = 0; rank < size; rank++) {
    ep_states.emplace_back(rank);
  }
}

void ServiceManager::add_service(std::unique_ptr<Service> serv) {
  servs.push_back(std::move(serv));
}

int ServiceManager::deserialize(EndpointState *estate) {
  char len_buf[sizeof(stream_len_t)];
  char sid_buf[sizeof(stream_sid_t)];
  char term_buf[sizeof(stream_term_t)];
  while (1) {
    switch (estate->sstate) {
      case StreamState::LEN:
        if (estate->ss.rdbuf()->in_avail() < sizeof(stream_len_t)) {
          return 0;
        }
        estate->ss.readsome(len_buf, sizeof(stream_len_t));
        estate->len = *(stream_len_t *)len_buf;
        estate->sstate = StreamState::SID;
        break;
      case StreamState::SID:
        if (estate->ss.rdbuf()->in_avail() < sizeof(stream_sid_t)) {
          return 0;
        }
        estate->ss.readsome(sid_buf, sizeof(stream_sid_t));
        estate->sid = *(stream_sid_t *)sid_buf;
        estate->sstate = StreamState::CONTENT;
        break;
      case StreamState::CONTENT:
        if (estate->ss.rdbuf()->in_avail() < estate->len) {
          return 0;
        }
        estate->sstate = StreamState::TERM;
        return 1;
      case StreamState::TERM:
        if (estate->ss.rdbuf()->in_avail() < sizeof(stream_term_t)) {
          return 0;
        }
        estate->ss.readsome(term_buf, sizeof(stream_term_t));
        CHECK(*(stream_term_t *)term_buf == TERM);
        estate->sstate = StreamState::LEN;
        break;
    }
  }
}

void ServiceManager::run(Communicator *comm, ServiceManager *self) {
  size_t length;
  ucs_status_ptr_t stat;
  while (!self->shutdown) {
    for (size_t srcrank = 0; srcrank < self->size; srcrank++) {
      if (srcrank == self->rank) continue;
      EndpointState *estate = &self->ep_states[srcrank];
      comm->get_data(srcrank, &estate->ss);
      while (deserialize(estate)) {
        self->servs[estate->sid]->recv(comm, estate);
      }
    }
    for (auto &serv: self->servs) {
      serv->progress(comm);
    }
    comm->progress();
  }
}

}
}