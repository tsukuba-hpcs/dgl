#include <gtest/gtest.h>
#include <dgl/graph.h>
#include <dgl/runtime/object.h>
#include "../src/distributedv2/dataloader.h"

using namespace dgl::distributedv2;
using namespace dmlc::moodycamel;

class FeatLoaderTest : public ::testing::Test {
protected:
  std::vector<int> _feat0, _feat1;
  dgl::NDArray feat0, feat1;
  Communicator comm0, comm1;
  ServiceManager sm0, sm1;
  std::queue<blocks_with_label_t> input0, input1;
  BlockingConcurrentQueue<blocks_with_feat_t> output0, output1;
  FeatLoaderTest()
  : comm0(0, 2, 100)
  , comm1(1, 2, 100)
  , sm0(0, 2, &comm0)
  , sm1(1, 2, &comm1)
  {}
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

TEST_F(FeatLoaderTest, TEST1) {
  {
    _feat0 = std::vector<int>{0,1,2,3,4};
    _feat1 = std::vector<int>{5,6,7,8,9};
    feat0 = dgl::NDArray::FromVector<int>(_feat0);
    feat1 = dgl::NDArray::FromVector<int>(_feat1);
    feat_loader_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 10,
      .local_feats = feat0,
    };
    feat_loader_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 10,
      .local_feats = feat1,
    };
    auto loader0 = 
      std::unique_ptr<FeatLoader>(new FeatLoader(std::move(arg0), &input0, &output0));
    auto loader1 = 
      std::unique_ptr<FeatLoader>(new FeatLoader(std::move(arg1), &input1, &output1));
    sm0.add_rma_service(std::move(loader0));
    sm1.add_rma_service(std::move(loader1));
    create_ep();
    mem_map();
  }
  std::vector<node_id_t> src_nodes{1, 6};
  blocks_with_label_t item = {
    .blocks = std::vector<dgl::HeteroGraphPtr>{}
  , .labels = dgl::NDArray::FromVector(std::vector<int>{0, 0})
  , .input_nodes = src_nodes,
  };
  input0.push(std::move(item));
  blocks_with_feat_t out;
  while (!output0.try_dequeue(out)) {
    sm0.progress();
    sm1.progress();
  }
  ASSERT_EQ(out.feats->ndim, 2);
  ASSERT_EQ(out.feats->shape[0], 2);
  ASSERT_EQ(out.feats->shape[1], 1);
  ASSERT_EQ(out.feats->dtype.bits, sizeof(int) * 8);
  int ret1 = *(int *)PTR_BYTE_OFFSET(out.feats->data, 0);
  int ret6 = *(int *)PTR_BYTE_OFFSET(out.feats->data, sizeof(int));
  ASSERT_EQ(ret1, 1);
  ASSERT_EQ(ret6, 6);
}