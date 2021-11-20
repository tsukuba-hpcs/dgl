#include "service.h"

#include <ucp/api/ucp.h>
#include <sstream>

namespace dgl {
namespace ucx {
namespace service {

GraphServer::GraphServer(GraphRef g)
  : local_graph(g) {
}

void GraphServer::recv(context::Context *ctx, EndpointState *estate) {

}

void GraphServer::tick(context::Context *ctx) {

}

ServiceManager::ServiceManager(int size)
  : shutdown(false) {
  for(int rank = 0; rank < size; rank++) {
    ep_states.emplace_back(rank);
  }
}

void ServiceManager::add_service(std::unique_ptr<Service> serv) {
  servs.push_back(std::move(serv));
}

int ServiceManager::deserialize(EndpointState *estate) {
  char buf[BUFFER_LEN];
  while (1) {
    switch (estate->sstate) {
      case StreamState::LEN:
        if (estate->ss.rdbuf()->in_avail() < sizeof(stream_len_t)) {
          return 0;
        }
        estate->ss.readsome(buf, sizeof(stream_len_t));
        estate->len = *(stream_len_t *)buf;
        estate->sstate = StreamState::SID;
        break;
      case StreamState::SID:
        if (estate->ss.rdbuf()->in_avail() < sizeof(stream_sid_t)) {
          return 0;
        }
        estate->ss.readsome(buf, sizeof(stream_sid_t));
        estate->sid = *(stream_sid_t *)buf;
        estate->sstate = StreamState::CONTENT;
        break;
      case StreamState::CONTENT:
        if (estate->ss.rdbuf()->in_avail() < estate->len) {
          return 0;
        }
        estate->sstate = StreamState::PARITY;
        return 1;
      case StreamState::PARITY:
        if (estate->ss.rdbuf()->in_avail() < sizeof(stream_parity_t)) {
          return 0;
        }
        estate->ss.readsome(buf, sizeof(stream_parity_t));
        CHECK(*(stream_parity_t *)buf == PARITY);
        estate->sstate = StreamState::LEN;
        break;
    }
  }
}

void ServiceManager::run(context::Context *ctx) {
  size_t length;
  ucs_status_ptr_t stat;
  while (!ctx->manager.shutdown) {
    for (size_t rank = 0; rank < ctx->eps.size(); rank++) {
      if (rank == ctx->rank) continue;
      stat = ucp_stream_recv_data_nb(ctx->eps[rank], &length);
      if (length == 0 || stat == NULL) continue;
      EndpointState *estate = &ctx->manager.ep_states[rank];
      estate->ss.write(reinterpret_cast<const char*>(stat), length);
      ucp_stream_data_release(ctx->eps[rank], stat);
      while (deserialize(estate)) {
        ctx->manager.servs[estate->sid]->recv(ctx, estate);
      }
    }
    for (auto &serv: ctx->manager.servs) {
      serv->tick(ctx);
    }
    ucp_worker_progress(ctx->ucp_worker);
  }
}

}
}
}