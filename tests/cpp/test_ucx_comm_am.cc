#include <gtest/gtest.h>
#include <string>
#include <memory>
#include <chrono>
#include "../src/distributedv2/comm.h"

using namespace dgl::distributedv2;

class CommAMTest : public ::testing::Test {
protected:
  Communicator comm0, comm1;
  CommAMTest() : comm0(0, 2, 100), comm1(1, 2, 100) {}
  void create_ep() {
    auto p0 = comm0.create_workers();
    auto p1 = comm1.create_workers();
    std::vector<char> addrs(p0.second + p1.second);
    std::memcpy(&addrs[0], p0.first, p0.second);
    std::memcpy(&addrs[p0.second], p1.first, p1.second);
    comm0.create_endpoints(&addrs[0], addrs.size());
    comm1.create_endpoints(&addrs[0], addrs.size());
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

TEST_F(CommAMTest, PING) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  unsigned id;
  id = comm0.add_am_handler(&ctx, recv_cb);
  ASSERT_EQ(id, 0);
  id = comm1.add_am_handler(NULL, recv_cb);
  ASSERT_EQ(id, 0);
  create_ep();
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
  std::strcpy((char *)data.get(), "Hello, world");
  comm1.am_post(0, id, std::move(data), sizeof("Hello, world"));
  comm1.progress();
  for (int trial = 0; trial < 10000; trial++) {
    comm0.progress();
    comm1.progress();
  }
  ASSERT_EQ(ctx.count, 1);
  ASSERT_EQ(ctx.len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.buf, "Hello, world");
}

TEST_F(CommAMTest, PING_MULTI) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  comm0.add_am_handler(&ctx, recv_cb);
  comm1.add_am_handler(NULL, recv_cb);
  create_ep();
  for (size_t idx = 0; idx < 100; idx++) {
    std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
    std::strcpy((char *)data.get(), "Hello, world");
    comm1.am_post(0, 0, std::move(data), sizeof("Hello, world"));
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

TEST_F(CommAMTest, REUSE_POOL) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  comm0.add_am_handler(&ctx, recv_cb);
  comm1.add_am_handler(NULL, recv_cb);
  create_ep();
  size_t req_cnt = 0;
  while (ctx.count < 100000) {
    comm0.progress();
    comm1.progress();
    if (req_cnt < 100000 && req_cnt - ctx.count < 100) {
      req_cnt++;
      std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
      std::strcpy((char *)data.get(), "Hello, world");
      comm1.am_post(0, 0, std::move(data), sizeof("Hello, world"));
    }
  }
  ASSERT_EQ(ctx.count, 100000);
  ASSERT_EQ(ctx.count, req_cnt);
  ASSERT_EQ(ctx.len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.buf, "Hello, world");
}

TEST_F(CommAMTest, REUSE_POOL_2) {
  comm_test_ctx ctx {
    .len = 0,
    .buf = NULL,
    .count = 0};
  comm0.add_am_handler(&ctx, recv_cb);
  comm1.add_am_handler(NULL, recv_cb);
  create_ep();
  size_t req_cnt = 0;
  while (ctx.count < 100000) {
    comm0.progress();
    comm1.progress();
    if (req_cnt < 100000 && (req_cnt - ctx.count + MAX_IOV_CNT) / MAX_IOV_CNT < 100) {
      req_cnt += MAX_IOV_CNT;
      for (uint8_t idx = 0; idx < MAX_IOV_CNT; idx++) {
        std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof("Hello, world")]);
        std::strcpy((char *)data.get(), "Hello, world");
        comm1.am_post(0, 0, std::move(data), sizeof("Hello, world"));
      }
    }
  }
  ASSERT_EQ(ctx.count, 100000);
  ASSERT_EQ(ctx.count, req_cnt);
  ASSERT_EQ(ctx.len, sizeof("Hello, world"));
  ASSERT_STREQ((char*)ctx.buf, "Hello, world");
}