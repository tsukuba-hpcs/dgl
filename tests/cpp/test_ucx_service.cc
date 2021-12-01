#include <gtest/gtest.h>
#include <dgl/graph.h>
#include <dgl/runtime/object.h>
#include "../src/distributedv2/context.h"
#include "../src/distributedv2/service.h"

using namespace dgl::distributedv2;


class ServTest : public ::testing::Test {
protected:
  Communicator comm0, comm1;
  ServiceManager sm0,sm1;
  std::queue<std::vector<uint64_t>> input0, input1;
  std::queue<std::vector<block_t>> output0, output1;
  ServTest()
  :
    comm0(0, 2, 100)
  , comm1(1, 2, 100)
  , sm0(0, 2, &comm0)
  , sm1(1, 2, &comm1)
  {
    int addrlen = comm0.get_workerlen();
    std::string addrs(addrlen * 2, (char)0);
    std::memcpy(&addrs[0], comm0.get_workeraddr(), addrlen);
    std::memcpy(&addrs[addrlen], comm1.get_workeraddr(), addrlen);
    comm0.create_endpoints(addrs);
    comm1.create_endpoints(addrs);
  }
};

TEST_F(ServTest, TEST1) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{4,5},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{0,0},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{3,2,1},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{4,4,5},64));
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 6,
      .num_layers = 2,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(6, edge0_src, edge0_dst)),
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 6,
      .num_layers = 2,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(6, edge1_src, edge1_dst)),
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_service(std::move(sampler0));
    sm1.add_service(std::move(sampler1));
  }

  std::vector<uint64_t> seeds{0};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 2);
  ASSERT_EQ(blocks[0].size(), 2);
  ASSERT_EQ(blocks[1].size(), 3);
  ASSERT_EQ(blocks[0][0].src, 4);
  ASSERT_EQ(blocks[0][0].dst, 0);
  ASSERT_EQ(blocks[0][1].src, 5);
  ASSERT_EQ(blocks[0][1].dst, 0);
  ASSERT_EQ(blocks[1][0].src, 3);
  ASSERT_EQ(blocks[1][0].dst, 4);
  ASSERT_EQ(blocks[1][1].src, 2);
  ASSERT_EQ(blocks[1][1].dst, 4);
  ASSERT_EQ(blocks[1][2].src, 1);
  ASSERT_EQ(blocks[1][2].dst, 5);
}