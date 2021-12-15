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
  std::queue<seed_with_blocks_t> input0, input1;
  ConcurrentQueue<seed_with_feat_t> output0, output1;
  FeatLoaderTest()
  : comm0(0, 2, 100)
  , comm1(1, 2, 100)
  , sm0(0, 2, &comm0)
  , sm1(1, 2, &comm1)
  {
    auto p0 = comm0.get_workeraddr();
    auto p1 = comm1.get_workeraddr();
    std::string addrs(p0.second + p1.second, (char)0);
    std::memcpy(&addrs[0], p0.first, p0.second);
    std::memcpy(&addrs[p0.second], p1.first, p1.second);
    comm0.create_endpoints(addrs);
    comm1.create_endpoints(addrs);
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
    rma_serv_ret_t ret0, ret1;
    ret0 = sm0.add_rma_service(std::move(loader0));
    ret1 = sm1.add_rma_service(std::move(loader1));
    std::vector<char> rkey_buf(ret0.rkey_buf_len + ret1.rkey_buf_len);
    std::memcpy(&rkey_buf[0], ret0.rkey_buf, ret0.rkey_buf_len);
    std::memcpy(&rkey_buf[ret0.rkey_buf_len], ret1.rkey_buf, ret1.rkey_buf_len);
    fprintf(stderr, "ret0.address=%p\n", ret0.address);
    std::vector<char> address(sizeof(void *) * 2);
    std::memcpy(&address[0], &ret0.address, sizeof(void *));
    std::memcpy(&address[sizeof(void *)], &ret1.address, sizeof(void *));
    sm0.setup_rma_service(ret0.rma_id, &rkey_buf[0], rkey_buf.size(), &address[0], address.size());
    sm1.setup_rma_service(ret1.rma_id, &rkey_buf[0], rkey_buf.size(), &address[0], address.size());
  }
  std::vector<dgl::dgl_id_t> src_nodes{1, 6};
  seed_with_blocks_t item(
    std::vector<dgl::dgl_id_t>{}
  , dgl::NDArray::FromVector(std::vector<int>{0})
  , std::vector<block_t>{block_t{
      .edges = std::vector<edge_elem_t>{}
    , .src_nodes = src_nodes}}
  );
  ASSERT_EQ(item.blocks.size(), 1);
  input0.push(std::move(item));
  
  seed_with_feat_t out;
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