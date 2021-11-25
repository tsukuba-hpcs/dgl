#include <gtest/gtest.h>
#include <string>
#include <memory>
#include <chrono>
#include "../src/distributedv2/context.h"

using namespace dgl::distributedv2;

class CommTest : public ::testing::Test {
protected:
  Communicator comm0, comm1;
  CommTest() : comm0(0, 2, 100), comm1(1, 2, 100) {
    int addrlen = comm0.get_workerlen();
    std::string addrs(addrlen * 2, (char)0);
    std::memcpy(&addrs[0], comm0.get_workeraddr(), addrlen);
    std::memcpy(&addrs[addrlen], comm1.get_workeraddr(), addrlen);
    comm0.create_endpoints(addrs);
    comm1.create_endpoints(addrs);
  }
};

struct comm_test_ctx {
  bool received;
  uint8_t iov_cnt;
  size_t first_len;
  void *first_buf;
};

static void recv_cb(void *arg, comm_iov_t *iov, uint8_t iov_cnt) {
  comm_test_ctx *ctx = (comm_test_ctx *)(arg);
  ASSERT_TRUE(iov_cnt > 0);
  ctx->received = true;
  ctx->iov_cnt = iov_cnt;
  ctx->first_len = iov[0].length;
  ctx->first_buf = std::malloc(iov[0].length);
  std::memcpy(ctx->first_buf, iov[0].buffer, iov[0].length);
}

TEST_F(CommTest, HELLO) {
  comm_test_ctx ctx {.received = false, .iov_cnt = 0, .first_len = 0, .first_buf = NULL};
  unsigned id;
  id = comm0.add_recv_handler(&ctx, recv_cb);
  ASSERT_EQ(id, 0);
  id = comm1.add_recv_handler(NULL, recv_cb);
  ASSERT_EQ(id, 0);
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
  std::strcpy((char *)data.get(), "Hello, world");
  comm1.append(0, id, std::move(data), sizeof("Hello, world"));
  comm1.progress();
  for (int trial = 0; trial < 10000; trial++) {
    comm0.progress();
    comm1.progress();
  }
  ASSERT_TRUE(ctx.received);
  ASSERT_EQ(ctx.iov_cnt, 1);
  ASSERT_EQ(ctx.first_len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.first_buf, "Hello, world");
}

