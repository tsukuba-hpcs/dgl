#include <gtest/gtest.h>
#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include "../src/distributedv2/context.h"

using namespace dgl::distributedv2;

class CommMemTest : public ::testing::Test {
protected:
  std::vector<char> buf0, buf1;
  Communicator comm0, comm1;
  CommMemTest() : comm0(0, 2, 100), comm1(1, 2, 100) {
    auto p0 = comm0.get_workeraddr();
    auto p1 = comm1.get_workeraddr();
    std::string addrs(p0.second + p1.second, (char)0);
    std::memcpy(&addrs[0], p0.first, p0.second);
    std::memcpy(&addrs[p0.second], p1.first, p1.second);
    comm0.create_endpoints(addrs);
    comm1.create_endpoints(addrs);
    buf0.resize(sizeof("HELLO"));
    std::strcpy(buf0.data(), "HELLO");
    buf1.resize(sizeof("WORLD"));
    std::strcpy(buf1.data(), "WORLD");
    unsigned id = comm0.add_rma_handler(buf0.data(), sizeof("HELLO"));
    CHECK(id == 0);
    CHECK(comm1.add_rma_handler(buf1.data(), sizeof("WORLD")) == id);
    auto r0 = comm0.get_rkey_buf(id);
    auto r1 = comm1.get_rkey_buf(id);
    CHECK(r0.second == r1.second);
    std::vector<char> rkey_buf(r0.second + r1.second);
    std::memcpy(&rkey_buf[0], r0.first, r0.second);
    std::memcpy(&rkey_buf[r0.second], r1.first, r1.second);
    comm0.create_rkey(id, rkey_buf.data(), rkey_buf.size());
    comm1.create_rkey(id, rkey_buf.data(), rkey_buf.size());
    std::vector<char> address(sizeof(uint64_t) * 2);
    std::memcpy(&address[0], buf0.data(), sizeof(uint64_t));
    std::memcpy(&address[sizeof(uint64_t)], buf1.data(), sizeof(uint64_t));
    comm0.set_buffer_addr(id, address.data(), address.size());
    comm1.set_buffer_addr(id, address.data(), address.size());
  }
};

TEST_F(CommMemTest, HELLO) {
  fprintf(stderr, "HELLO\n");
}