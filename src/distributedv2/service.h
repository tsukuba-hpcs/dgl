/*!
 *  Copyright (c) 2021 by Contributors
 * \file distributedv2/service.h
 * \brief headers for distv2 service.
 */

#ifndef DGL_DISTV2_SERVICE_H_
#define DGL_DISTV2_SERVICE_H_

#include <stdlib.h>
#include <dgl/graph.h>
#include <memory>
#include <vector>
#include <atomic>
#include <sstream>

#include "comm.h"

namespace dgl {
namespace distributedv2 {

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
  , sstate(StreamState::LEN)
  , sid(0)
  , len(0) {
  }
};

class Service {
public:
  virtual void recv(Communicator *comm, EndpointState *estate) = 0;
  virtual void progress(Communicator *comm) = 0;
};

class GraphServer: Service {
  GraphRef local_graph;
public:
  GraphServer(GraphRef g);
  void recv(Communicator *comm, EndpointState *estate);
  void progress(Communicator *comm);
};

class ServiceManager {
  std::vector<std::unique_ptr<Service>> servs;
  std::vector<EndpointState> ep_states;
  std::atomic_bool shutdown;
  int rank, size;
public:
  using stream_len_t = uint16_t;
  using stream_sid_t = char;
  using stream_term_t = char;
  const static stream_term_t TERM = 0x77;
  const static size_t MAX_STREAM_LENGTH = 1<<28;
  ServiceManager(int rank, int size);
  void add_service(std::unique_ptr<Service> serv);
  static std::string serialize(stream_sid_t sid, const char *data, stream_len_t len);
  static int deserialize(EndpointState *estate);
  static void run(Communicator *comm, ServiceManager *self);
};

}
}

#endif