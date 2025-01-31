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
#ifdef DGL_USE_NVTX
#include <nvToolsExt.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif  // DGL_USE_NVTX


#include "comm.h"

namespace dgl {
namespace distributedv2 {

#define DGL_FG_THREAD_AFFINITY "DGL_FG_THREAD_AFFINITY"
#define DGL_BG_THREAD_AFFINITY "DGL_BG_THREAD_AFFINITY"

class Service {
public:
  virtual unsigned progress(Communicator *comm) = 0;
};

class StubService: public Service {
public:
  unsigned stub_id;
  virtual unsigned progress() = 0;
  unsigned progress(Communicator *comm) {
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
  virtual void rma_read_cb(Communicator *comm, uint64_t req_id) = 0;
};

struct sm_cb_arg_t {
  Service *serv;
  Communicator *comm;
  sm_cb_arg_t(Service *serv, Communicator *comm)
  : serv(serv), comm(comm) {};
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
  static void rma_recv_cb(void *arg, uint64_t req_id);
  // for debugging
  size_t progress_counter = 0;
  size_t act_counter = 0;
  static constexpr size_t YIELD_THRESHOLD = (1<<16);
public:
  ServiceManager(int rank, int size, Communicator *comm);
  void add_stub_service(std::unique_ptr<StubService> &&serv);
  void add_am_service(std::unique_ptr<AMService> &&serv);
  void add_rma_service(std::unique_ptr<RMAService> &&serv);
  rma_mem_ret_t map_rma_service();
  void prepare_rma_service(void *rkeybuf, size_t rkeybuf_len, void *address, size_t address_len);
  void progress();
  void launch();
  void terminate();
  static void run(ServiceManager *self);
};

}
}

#endif