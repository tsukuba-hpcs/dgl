/*!
 *  Copyright (c) 2021 by Contributors
 * \file distributedv2/comm.h
 * \brief headers for distv2 comm.
 */

#ifndef DGL_DISTV2_COMM_H_
#define DGL_DISTV2_COMM_H_

#include <ucp/api/ucp.h>
#include <dgl/runtime/object.h>

#include <string>

namespace dgl {
namespace distributedv2 {

class Communicator {
  int rank;
  int size;
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  ucp_address_t* addr;
  size_t addrlen;
  std::vector<ucp_ep_h> eps;
public:
  Communicator(int rank, int size);
  ~Communicator();
  ucp_address_t* get_workeraddr();
  int get_workerlen();
  void create_endpoints(std::string addrs);
  void get_data(int rank, std::stringstream *ss);
  void progress();
};

}
}

#endif