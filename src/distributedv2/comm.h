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


typedef void (*comm_am_cb_t)(void *arg, const void *buffer, size_t length);

typedef void (*comm_rma_cb_t)(void *arg, uint64_t req_id, void *address);

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

struct am_handler_t {
  void *arg;
  comm_am_cb_t cb;
};

struct rma_handler_t {
  void *arg;
  comm_rma_cb_t cb;
  ucp_mem_h mem;
  void *rkey_buf;
  size_t rkey_buf_len;
  std::vector<ucp_rkey_h> rkey;
  std::vector<uint64_t> address;
  rma_handler_t(void *arg, comm_rma_cb_t cb, ucp_mem_h mem, void *rkey_buf, size_t rkey_buf_len)
  : arg(arg), cb(cb), mem(mem), rkey_buf(rkey_buf), rkey_buf_len(rkey_buf_len) {}
};

struct rma_pool_item_t {
  bool used;
  rma_handler_t *handler;
  uint64_t req_id;
  void *address;
  rma_pool_item_t(rma_handler_t *handler);
  void release();
};

class RmaPool {
  std::vector<rma_pool_item_t> buffer;
  size_t cursor;
public:
  RmaPool(size_t length);
  int alloc(rma_pool_item_t** item, size_t req_id, void *address, rma_handler_t *handler);
};

class Communicator: public runtime::Object {
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  ucp_address_t* addr;
  size_t addrlen;
  std::vector<ucp_ep_h> eps;
  // for Active Message
  IovPool am_pool;
  std::vector<am_handler_t> recv_handlers;
  std::vector<std::vector<iov_pool_item_t*>> chunks;
  static ucs_status_t recv_cb(
    void *arg,
    const void *header,
    size_t header_length,
    void *data, size_t length,
    const ucp_am_recv_param_t *param);
  static void send_cb(void *request, ucs_status_t status, void *user_data);
  void am_send(int rank, unsigned id, iov_pool_item_t *chunk);
  // for Remote Memory Access
  RmaPool rma_pool;
  std::vector<rma_handler_t> mem_handlers;
  static void read_cb(void *request, ucs_status_t status, void *user_data);
public:
  const int rank;
  const int size;
  Communicator(int rank, int size, size_t buffer_len = (1<<20));
  ~Communicator();
  // for Endpoints
  std::pair<ucp_address_t*, int> get_workeraddr();
  void create_endpoints(std::string addrs);
  // for Active Message
  unsigned add_am_handler(void *arg, comm_am_cb_t cb);
  void am_post(int destrank, unsigned am_id, std::unique_ptr<uint8_t[]> &&data, size_t length);
  // for Remote Memory Access
  unsigned add_rma_handler(void *buffer, size_t length, void *arg, comm_rma_cb_t cb);
  std::pair<void*, size_t> get_rkey_buf(unsigned rma_id);
  void create_rkey(unsigned rma_id, const void *buffer, size_t length);
  void set_buffer_addr(unsigned rma_id, const intptr_t buffer, size_t length);
  void rma_read(int destrank, unsigned rma_id, uint64_t req_id, void *buffer, uint64_t offset, size_t length);
  // for Progress
  void progress();
  static constexpr const char* _type_key = "distributedv2.Communicator";
  DGL_DECLARE_OBJECT_TYPE_INFO(Communicator, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(CommunicatorRef, Communicator);

}
}

#endif