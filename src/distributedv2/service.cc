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
  int length;
  int pos;
  while (1) {
    switch (estate->sstate) {
      case StreamState::LEN:
        length = estate->ss.rdbuf()->sgetn(len_buf, sizeof(stream_len_t));
        if (length < sizeof(stream_len_t)) {
          pos = estate->ss.tellg() - length;
          estate->ss.seekg(pos);
          return 0;
        }
        estate->len = *(stream_len_t *)len_buf;
        estate->sstate = StreamState::SID;
        break;
      case StreamState::SID:
        length = estate->ss.rdbuf()->sgetn(sid_buf, sizeof(stream_sid_t));
        if (length < sizeof(stream_sid_t)) {
          pos = estate->ss.tellg() - length;
          estate->ss.seekg(pos);
          return 0;
        }
        estate->sid = *(stream_sid_t *)sid_buf;
        estate->sstate = StreamState::CONTENT;
        break;
      case StreamState::CONTENT:
        length = estate->ss.rdbuf()->sgetn(sid_buf, estate->len);
        pos = estate->ss.tellg() - length;
        estate->ss.seekg(pos);
        if (length < estate->len) {
          return 0;
        }
        estate->sstate = StreamState::TERM;
        return 1;
      case StreamState::TERM:
        length = estate->ss.rdbuf()->sgetn(term_buf, sizeof(stream_term_t));
        if (length < sizeof(stream_term_t)) {
          pos = estate->ss.tellg() - length;
          estate->ss.seekg(pos);
          return 0;
        }
        CHECK(*(stream_term_t *)term_buf == TERM);
        pos = estate->ss.tellg();
        estate->sstate = StreamState::LEN;
        if (pos > MAX_STREAM_LENGTH) {
          // Garbage Collection for Stream Buffer
          std::string str(estate->ss.str());
          estate->ss = std::stringstream();
          estate->ss.write(str.c_str(), str.size());
        }
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