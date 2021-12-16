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
#include <thread>


#include "comm.h"

namespace dgl {
namespace distributedv2 {

class Service {
public:
  virtual void progress(Communicator *comm) = 0;
};

class StubService: public Service {
public:
  unsigned stub_id;
  virtual void progress() = 0;
  void progress(Communicator *comm) {
    progress();
  }
};

class AMService: public Service {
public:
  unsigned am_id;
  virtual void am_recv(Communicator *comm, const void *buffer, size_t length) = 0;
};

class RMAService: public Service {
public:
  unsigned rma_id;
  virtual std::pair<void *, size_t> served_buffer() = 0;
  virtual void rma_read_cb(Communicator *comm, uint64_t req_id, void *buffer) = 0;
};

struct sm_cb_arg_t {
  Service *serv;
  Communicator *comm;
  sm_cb_arg_t(Service *serv, Communicator *comm)
  : serv(serv), comm(comm) {};
};

struct rma_serv_ret_t {
  unsigned rma_id;
  void *rkey_buf;
  size_t rkey_buf_len;
  void *address;
};

class ServiceManager {
  static constexpr size_t MAX_SERVICE_LEN = 10;
  std::vector<std::unique_ptr<Service>> servs;
  std::vector<sm_cb_arg_t> args;
  Communicator *comm;
  std::atomic_bool shutdown;
  int rank, size;
  std::thread progress_thread;
  static void am_recv_cb(void *arg, const void *buffer, size_t length);
  static void rma_recv_cb(void *arg, uint64_t req_id, void *address);
public:
  ServiceManager(int rank, int size, Communicator *comm);
  void add_stub_service(std::unique_ptr<StubService> &&serv);
  void add_am_service(std::unique_ptr<AMService> &&serv);
  rma_serv_ret_t add_rma_service(std::unique_ptr<RMAService> &&serv);
  void setup_rma_service(unsigned rma_id, void *rkey_bufs, size_t rkey_buf_len, void *address, size_t addr_len);
  void progress();
  void launch();
  void terminate();
  static void run(ServiceManager *self);
};

}
}

#endif