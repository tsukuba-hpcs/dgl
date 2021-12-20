#include <gtest/gtest.h>
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include "../src/distributedv2/comm.h"

using namespace dgl::distributedv2;

class CommRMATest : public ::testing::Test {
protected:
  std::vector<char> buf0, buf1, recv_buf;
  Communicator comm0, comm1;
  CommRMATest() : comm0(0, 2, 100), comm1(1, 2, 100) {}
  void create_ep() {
    auto p0 = comm0.create_workers();
    auto p1 = comm1.create_workers();
    std::vector<char> addrs(p0.second + p1.second);
    std::memcpy(&addrs[0], p0.first, p0.second);
    std::memcpy(&addrs[p0.second], p1.first, p1.second);
    comm0.create_endpoints(&addrs[0], addrs.size());
    comm1.create_endpoints(&addrs[0], addrs.size());
  }
  void mem_map() {
    auto r0 = comm0.rma_mem_map();
    auto r1 = comm1.rma_mem_map();
    ASSERT_EQ(r0.address_len, r1.address_len);
    ASSERT_EQ(r0.address_len, sizeof(uint64_t));
    ASSERT_EQ(r0.rkeybuf_len, r1.rkeybuf_len);
    std::vector<char> rkeybuf(r0.rkeybuf_len + r1.rkeybuf_len);
    std::memcpy(&rkeybuf[0], r0.rkeybuf, r0.rkeybuf_len);
    std::memcpy(&rkeybuf[r0.rkeybuf_len], r1.rkeybuf, r1.rkeybuf_len);
    std::vector<char> address(r0.address_len + r1.address_len);
    std::memcpy(&address[0], r0.address, r0.address_len);
    std::memcpy(&address[r0.address_len], r1.address, r1.address_len);
    comm0.prepare_rma(&rkeybuf[0], rkeybuf.size(), &address[0], address.size());
    comm1.prepare_rma(&rkeybuf[0], rkeybuf.size(), &address[0], address.size());
  }
};


static void recv_cb(void *arg, uint64_t req_id, void *address) {
  *((bool *)arg) = true;
}

TEST_F(CommRMATest, HELLO) {
  buf0.resize(sizeof("HELLO"));
  std::strcpy(buf0.data(), "HELLO");
  buf1.resize(sizeof("WORLD"));
  recv_buf.resize(sizeof("WORLD"));
  std::strcpy(buf1.data(), "WORLD");
  bool finished = false;
  fprintf(stderr, "buf0.data()=%p\n", buf0.data());
  unsigned id = comm0.add_rma_handler(buf0.data(), sizeof("HELLO"), &finished, recv_cb);
  ASSERT_EQ(id, 0);
  ASSERT_EQ(comm1.add_rma_handler(buf1.data(), sizeof("WORLD"), NULL, recv_cb), id);
  create_ep();
  mem_map();
  comm0.rma_read(1, id, 0, &recv_buf[0], 0, recv_buf.size());
  while (!finished) {
    comm0.progress();
    comm1.progress();
  }
  ASSERT_STREQ(buf1.data(), recv_buf.data());
}