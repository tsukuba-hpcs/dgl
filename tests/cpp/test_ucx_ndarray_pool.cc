#include <gtest/gtest.h>
#include <dgl/graph.h>
#include <dgl/runtime/object.h>
#include "../src/distributedv2/dataloader.h"

using namespace dgl::distributedv2;

TEST(NDARRAY_POOL, HELLO) {
  NDArrayPool pool(1<<10);
  for (int trial=0; trial < 100000; trial++) {
    DLDataType dtype{kDLInt, 8 * sizeof(node_id_t), 1};
    dgl::NDArray nda = pool.alloc(std::vector<int64_t>{1}, std::move(dtype));
    CHECK_EQ(nda->shape[0], 1);
  }
}

TEST(NDARRAY_POOL, HELLO_2) {
  NDArrayPool pool(1<<11);
  for (int trial=0; trial < 100000; trial++) {
    DLDataType dtype{kDLInt, 8 * sizeof(node_id_t), 1};
    dgl::NDArray nda = pool.alloc(std::vector<int64_t>{16,16}, std::move(dtype));
    CHECK_EQ(nda->shape[0], 16);
    CHECK_EQ(nda->shape[1], 16);
  }
}

TEST(NDARRAY_POOL, HELLO_3) {
  NDArrayPool pool(1<<11);
  for (int trial=0; trial < 100000; trial++) {
    std::vector<dgl::NDArray> temp;
    for (int cnt = 0; cnt < 16; cnt++) {
      DLDataType dtype{kDLInt, 8 * sizeof(node_id_t), 1};
      dgl::NDArray nda = pool.alloc(std::vector<int64_t>{16}, std::move(dtype));
      temp.push_back(std::move(nda));
    }
  }
}

TEST(NDARRAY_POOL, HELLO_4) {
  NDArrayPool pool(1<<11);
  DLDataType dtype{kDLInt, 8 * sizeof(node_id_t), 1};
  dgl::NDArray nda = pool.alloc(std::vector<int64_t>{16}, std::move(dtype));

  for (int trial=0; trial < 100000; trial++) {
    std::vector<dgl::NDArray> temp;
    for (int cnt = 0; cnt < 15; cnt++) {
      DLDataType dtype{kDLInt, 8 * sizeof(node_id_t), 1};
      dgl::NDArray ndb = pool.alloc(std::vector<int64_t>{16}, std::move(dtype));
      temp.push_back(std::move(ndb));
    }
  }
}