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
  virtual void progress(Communicator *comm) = 0;
};

class AMService: public Service {
public:
  unsigned am_id;
  virtual void am_recv(Communicator *comm, const void *buffer, size_t length) = 0;
};

class RMAService: public Service {
public:
  unsigned rma_id;
  virtual void rma_read_cb(Communicator *comm) = 0;
};

struct sm_cb_arg_t {
  Service *serv;
  Communicator *comm;
  sm_cb_arg_t(Service *serv, Communicator *comm)
  : serv(serv), comm(comm) {};
};

class ServiceManager {
  std::vector<std::unique_ptr<Service>> servs;
  std::vector<sm_cb_arg_t> args;
  Communicator *comm;
  std::atomic_bool shutdown;
  int rank, size;
  static void am_recv_cb(void *arg, const void *buffer, size_t length);
public:
  ServiceManager(int rank, int size, Communicator *comm);
  void add_am_service(std::unique_ptr<AMService> &&serv);
  void add_rma_service(std::unique_ptr<RMAService> &&serv);
  void progress();
  static void run(ServiceManager *self);
};

}
}

#endif