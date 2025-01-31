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

typedef void (*comm_rma_cb_t)(void *arg, uint64_t req_id);

#define MAX_IOV_CNT 8

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
  ucp_worker_h worker;
  size_t worker_addr_len;
  std::vector<ucp_ep_h> eps;
};

struct rma_handler_t {
  void *arg;
  comm_rma_cb_t cb;
  ucp_worker_h worker;
  size_t worker_addr_len;
  std::vector<ucp_ep_h> eps;
  ucp_mem_h mem;
  void *address;
  size_t buffer_len;
  void *rkeybuf;
  size_t rkeybuf_len;
  std::vector<ucp_rkey_h> rkeys;
  std::vector<uint64_t> addresses;
  rma_handler_t(void *arg, comm_rma_cb_t cb, void *address, size_t buffer_len)
  : arg(arg), cb(cb), address(address), buffer_len(buffer_len) {}
};

struct rma_pool_item_t {
  bool used;
  rma_handler_t *handler;
  uint64_t req_id;
  size_t bulk_size;
  rma_pool_item_t(rma_handler_t *handler);
  void release();
};

class RmaPool {
  std::vector<rma_pool_item_t> buffer;
  size_t cursor;
public:
  RmaPool(size_t length);
  int alloc(rma_pool_item_t** item, size_t req_id, rma_handler_t *handler);
};

enum class CommState {
  INIT,
  WORKER_READY,
  EP_READY,
  MEM_MAPPED,
  READY,
};

struct rma_mem_ret_t {
  void *rkeybuf;
  int rkeybuf_len;
  void *address;
  int address_len;
};

class Communicator: public runtime::Object {
  CommState state;
  std::vector<uint8_t> worker_addrs;
  static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status);
  // for Active Message
  ucp_context_h am_context;
  IovPool am_pool;
  std::vector<am_handler_t> am_handlers;
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
  ucp_context_h rma_context;
  RmaPool rma_pool;
  std::vector<rma_handler_t> rma_handlers;
  std::vector<uint8_t> rma_rkeybuf;
  std::vector<uint8_t> rma_address;
  static void read_cb(void *request, ucs_status_t status, void *user_data);
public:
  const int rank;
  const int size;
  Communicator(int rank, int size, size_t am_buffer_len = (1<<16), size_t rma_buffer_len = (1<<5));
  ~Communicator();
  // for Endpoints
  std::pair<void*, int> create_workers();
  void create_endpoints(void *addrs, size_t length);
  // for Active Message
  unsigned add_am_handler(void *arg, comm_am_cb_t cb);
  void am_post(int destrank, unsigned am_id, std::unique_ptr<uint8_t[]> &&data, size_t length);
  // for Remote Memory Access
  unsigned add_rma_handler(void *buffer, size_t length, void *arg, comm_rma_cb_t cb);
  rma_mem_ret_t rma_mem_map();
  void prepare_rma(void *rkeybuf, size_t rkeybuf_len, void *address, size_t address_len);
  void rma_read(int destrank, unsigned rma_id, uint64_t req_id, void *buffer, uint64_t offset, size_t length);
  // for Progress
  unsigned progress();
  static constexpr const char* _type_key = "distributedv2.Communicator";
  DGL_DECLARE_OBJECT_TYPE_INFO(Communicator, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(CommunicatorRef, Communicator);

}
}

#endif