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


typedef void (*comm_cb_handler_t)(void *arg, const void *buffer, size_t length);

#define MAX_IOV_CNT 16

#define PTR_BYTE_OFFSET UCS_PTR_BYTE_OFFSET

struct iov_pool_item_t {
  bool used;
  uint8_t iov_cnt;
  ucp_dt_iov_t iov[MAX_IOV_CNT];
  static constexpr size_t HEADER_LEN = MAX_IOV_CNT * sizeof(size_t) + sizeof(uint8_t);
  uint8_t header[HEADER_LEN];
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
};

struct comm_mem_handler_t {
  ucp_mem_h mem;
  void *rkey_buf;
  size_t rkey_buf_len;
  std::vector<ucp_rkey_h> rkey;
  std::vector<uint64_t> address;
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
  std::vector<comm_mem_handler_t> mem_handlers;
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
  // for Endpoints
  std::pair<ucp_address_t*, int> get_workeraddr();
  void create_endpoints(std::string addrs);
  // for Active Message
  unsigned add_recv_handler(void *arg, comm_cb_handler_t cb);
  void post(int rank, unsigned id, std::unique_ptr<uint8_t[]> &&data, size_t length);
  // for Remote Memory Access
  unsigned register_mem(void *buffer, size_t length);
  std::pair<void*, size_t> get_rkey_buf(unsigned id);
  void create_rkey(unsigned id, const void *buffer, size_t length);
  void set_buffer_addr(unsigned id, const void *buffer, size_t length);
  // for Progress
  void progress();
};

}
}

#endif