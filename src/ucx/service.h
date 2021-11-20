/*!
 *  Copyright (c) 2021 by Contributors
 * \file ucx/context.h
 * \brief headers for UCX context.
 */

#ifndef DGL_UCX_SERVICE_H_
#define DGL_UCX_SERVICE_H_

#include <stdlib.h>
#include <dgl/graph.h>
#include <memory>
#include <vector>
#include <atomic>
#include <sstream>

#include "context.h"

namespace dgl {
namespace ucx {
namespace service {

class Service {
public:
  Service();
  ~Service() = default;
  virtual void recv(context::Context *ctx, EndpointState *estate) = 0;
  virtual void tick(context::Context *ctx) = 0;
};

class GraphServer: Service {
  GraphRef local_graph;
public:
  GraphServer(GraphRef g);
  void recv(context::Context *ctx, EndpointState *estate);
  void tick(context::Context *ctx);
};

enum StreamState {
  LEN = 0,     /* length    ,   2 byte */
  SID = 1,     /* service id,   1 byte */
  CONTENT = 2, /* content   , len byte */
  TERM = 3,    /* term      ,   1 byte */
};

struct EndpointState {
  int rank;
  StreamState sstate;
  size_t sid;
  size_t len;
  std::stringstream ss;
  EndpointState(int rank)
  : rank(rank)
  , sstate(StreamState::SID)
  , sid(0)
  , len(0) {
  }
};

class ServiceManager {
  using stream_len_t = uint16_t;
  using stream_sid_t = char;
  using stream_term_t = char;
  constexpr static int BUFFER_LEN = std::max(
    sizeof(stream_len_t),
    sizeof(stream_sid_t),
    sizeof(stream_term_t));
  constexpr static stream_term_t TERM = 0x77;
  std::vector<std::unique_ptr<Service>> servs;
  std::vector<EndpointState> ep_states;
  std::atomic_bool shutdown;
  static int deserialize(EndpointState *estate);
public:
  ServiceManager(int size);
  void add_service(std::unique_ptr<Service> serv);
  static void run(context::Context *ctx);
};

}
}
}

#endif