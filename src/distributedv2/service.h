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
#include <queue>
#include <unordered_map>


#include "comm.h"

namespace dgl {
namespace distributedv2 {

class Service {
public:
  unsigned sid;
  virtual void recv(Communicator *comm, const void *buffer, size_t length) = 0;
  virtual void progress(Communicator *comm) = 0;
};

struct sm_recv_cb_arg_t {
  Service *serv;
  Communicator *comm;
  sm_recv_cb_arg_t(Service *serv, Communicator *comm)
  : serv(std::move(serv)), comm(std::move(comm)) {};
};

class ServiceManager {
  std::vector<std::unique_ptr<Service>> servs;
  std::vector<sm_recv_cb_arg_t> args;
  Communicator *comm;
  std::atomic_bool shutdown;
  int rank, size;
  static void recv_cb(void *arg, const void *buffer, size_t length);
public:
  ServiceManager(int rank, int size, Communicator *comm);
  void add_service(std::unique_ptr<Service> &&serv);
  void progress();
  static void run(ServiceManager *self);
};

}
}

#endif