#include <gtest/gtest.h>
#include <string>
#include <memory>
#include "../src/distributedv2/context.h"

using namespace dgl::distributedv2;


static void recv_cb(void *arg, comm_iov_t *iov, uint8_t iov_cnt) {
  fprintf(stderr, "recv_cb: iov_cnt=%u iov[0].length = %zu", iov_cnt, iov[0].length);
}

TEST(COMM_TEST, HELLO) {
  Communicator comm0(0, 2), comm1(1, 2);
  int addrlen = comm0.get_workerlen();
  std::string addrs(addrlen * 2, (char)0);
  std::memcpy(&addrs[0], comm0.get_workeraddr(), addrlen);
  std::memcpy(&addrs[addrlen], comm1.get_workeraddr(), addrlen);
  comm0.create_endpoints(addrs);
  comm1.create_endpoints(addrs);

  unsigned id;
  id = comm0.add_recv_handler(NULL, recv_cb);
  comm1.add_recv_handler(NULL, recv_cb);
  ASSERT_EQ(id, 0);
  char *data = (char *)malloc(sizeof("Hello, world")); 
  std::strcpy(data, "Hello, world");
  comm1.append(0, id, data, 13);
  comm1.progress();
  for (int trial = 0; trial < 10; trial++) {
    comm0.progress();
  }
}

