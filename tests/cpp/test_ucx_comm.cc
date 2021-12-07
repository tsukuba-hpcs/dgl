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
    auto p0 = comm0.get_workeraddr();
    auto p1 = comm1.get_workeraddr();
    std::string addrs(p0.second + p1.second, (char)0);
    std::memcpy(&addrs[0], p0.first, p0.second);
    std::memcpy(&addrs[p0.second], p1.first, p1.second);
    comm0.create_endpoints(addrs);
    comm1.create_endpoints(addrs);
  }
};

struct comm_test_ctx {
  size_t len;
  void *buf;
  size_t count;
};

static void recv_cb(void *arg, const void *buffer, size_t length) {
  comm_test_ctx *ctx = (comm_test_ctx *)(arg);
  ctx->count++;
  ctx->len = length;
  if (ctx->buf == NULL) ctx->buf = std::malloc(length);
  std::memcpy(ctx->buf, buffer, length);
}

TEST_F(CommTest, PING) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  unsigned id;
  id = comm0.add_recv_handler(&ctx, recv_cb);
  ASSERT_EQ(id, 0);
  id = comm1.add_recv_handler(NULL, recv_cb);
  ASSERT_EQ(id, 0);
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
  std::strcpy((char *)data.get(), "Hello, world");
  comm1.post(0, id, std::move(data), sizeof("Hello, world"));
  comm1.progress();
  for (int trial = 0; trial < 10000; trial++) {
    comm0.progress();
    comm1.progress();
  }
  ASSERT_EQ(ctx.count, 1);
  ASSERT_EQ(ctx.len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.buf, "Hello, world");
}

TEST_F(CommTest, PING_MULTI) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  comm0.add_recv_handler(&ctx, recv_cb);
  comm1.add_recv_handler(NULL, recv_cb);
  for (size_t idx = 0; idx < 100; idx++) {
    std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
    std::strcpy((char *)data.get(), "Hello, world");
    comm1.post(0, 0, std::move(data), sizeof("Hello, world"));
    comm0.progress();
  }
  ASSERT_EQ(ctx.count, 0);
  for (int trial = 0; trial < 10000; trial++) {
    comm0.progress();
    comm1.progress();
  }
  ASSERT_EQ(ctx.count, 100);
  ASSERT_EQ(ctx.len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.buf, "Hello, world");
}

TEST_F(CommTest, REUSE_POOL) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  comm0.add_recv_handler(&ctx, recv_cb);
  comm1.add_recv_handler(NULL, recv_cb);
  size_t req_cnt = 0;
  while (ctx.count < 10000000) {
    comm0.progress();
    comm1.progress();
    if (req_cnt < 10000000 && req_cnt - ctx.count < 100) {
      req_cnt++;
      std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
      std::strcpy((char *)data.get(), "Hello, world");
      comm1.post(0, 0, std::move(data), sizeof("Hello, world"));
    }
  }
  ASSERT_EQ(ctx.count, 10000000);
  ASSERT_EQ(ctx.count, req_cnt);
  ASSERT_EQ(ctx.len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.buf, "Hello, world");
}

TEST_F(CommTest, REUSE_POOL_2) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  comm0.add_recv_handler(&ctx, recv_cb);
  comm1.add_recv_handler(NULL, recv_cb);
  size_t req_cnt = 0;
  while (ctx.count < 10000000) {
    comm0.progress();
    comm1.progress();
    if (req_cnt < 10000000 && (req_cnt - ctx.count + MAX_IOV_CNT) / MAX_IOV_CNT < 100) {
      req_cnt += MAX_IOV_CNT;
      for (uint8_t idx = 0; idx < MAX_IOV_CNT; idx++) {
        std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
        std::strcpy((char *)data.get(), "Hello, world");
        comm1.post(0, 0, std::move(data), sizeof("Hello, world"));
      }
    }
  }
  ASSERT_EQ(ctx.count, 10000000);
  ASSERT_EQ(ctx.count, req_cnt);
  ASSERT_EQ(ctx.len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.buf, "Hello, world");
}