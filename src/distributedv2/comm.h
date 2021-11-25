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
#include <cstring>

namespace dgl {
namespace distributedv2 {

typedef ucp_dt_iov_t comm_iov_t;

typedef void (*comm_cb_handler_t)(void *arg, comm_iov_t *iov, uint8_t iov_cnt);

#define MAX_IOV_CNT 16

struct iov_pool_item_t {
  bool used;
  uint8_t iov_cnt;
  comm_iov_t iov[MAX_IOV_CNT];
  uint8_t header[MAX_IOV_CNT * sizeof(size_t) + sizeof(uint8_t)];
  std::unique_ptr<uint8_t[]> data[MAX_IOV_CNT];
  iov_pool_item_t();
  void release();
  size_t fill_header();
  bool filled();
  bool empty();
  void append(std::unique_ptr<uint8_t[]> &&data, size_t length);
};

class IovPool {
std::vector<iov_pool_item_t> buffer;
size_t cursor;
public:
  IovPool(size_t length);
  int alloc(iov_pool_item_t** item);
};

struct comm_recv_handler_t {
  void *arg;
  comm_cb_handler_t cb;
  IovPool *pool;
};

class Communicator {
  int rank;
  int size;
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  ucp_address_t* addr;
  size_t addrlen;
  std::vector<ucp_ep_h> eps;
  IovPool pool;
  std::vector<comm_recv_handler_t> recv_handlers;
  std::vector<std::vector<iov_pool_item_t*>> chunks;
  static ucs_status_t recv_cb(
    void *arg,
    const void *header,
    size_t header_length,
    void *data, size_t length,
    const ucp_am_recv_param_t *param);
  static void send_cb(void *request, ucs_status_t status, void *user_data);
  void send(int rank, unsigned id, iov_pool_item_t *chunk);
public:
  Communicator(int rank, int size, size_t buffer_len = (1<<20));
  ~Communicator();
  ucp_address_t* get_workeraddr();
  int get_workerlen();
  unsigned add_recv_handler(void *arg, comm_cb_handler_t cb);
  void append(int rank, unsigned id, std::unique_ptr<uint8_t[]> &&data, size_t length);
  void create_endpoints(std::string addrs);
  void progress();
};

}
}

#endif