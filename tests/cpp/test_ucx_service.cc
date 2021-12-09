#include <gtest/gtest.h>
#include <dgl/graph.h>
#include <dgl/runtime/object.h>
#include "../src/distributedv2/context.h"
#include "../src/distributedv2/service.h"
#include "../src/distributedv2/neighbor.h"

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
    auto p0 = comm0.get_workeraddr();
    auto p1 = comm1.get_workeraddr();
    std::string addrs(p0.second + p1.second, (char)0);
    std::memcpy(&addrs[0], p0.first, p0.second);
    std::memcpy(&addrs[p0.second], p1.first, p1.second);
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
      .fanouts = NULL,
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 6,
      .num_layers = 2,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(6, edge1_src, edge1_dst)),
      .fanouts = NULL,
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
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
  ASSERT_EQ(blocks[1][0].src, 1);
  ASSERT_EQ(blocks[1][0].dst, 5);
  ASSERT_EQ(blocks[1][1].src, 2);
  ASSERT_EQ(blocks[1][1].dst, 4);
  ASSERT_EQ(blocks[1][2].src, 3);
  ASSERT_EQ(blocks[1][2].dst, 4);
}

TEST_F(ServTest, KARATE_CLUB_1) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,4,4,5,5,5,6},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{1,2,3,4,5,6,7,8,10,11,12,13,2,3,7,13,3,7,8,9,13,7,12,13,6,10,6,10,16,16},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,1,1,1,1,2,2,2,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{17,19,21,31,17,19,21,30,27,28,32,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33},64));
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 1,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge0_src, edge0_dst)),
      .fanouts = NULL,
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 1,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge1_src, edge1_dst)),
      .fanouts = NULL,
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
  }
  std::vector<uint64_t> seeds{33};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 1);
  ASSERT_EQ(blocks[0].size(), 17);
  ASSERT_EQ(blocks[0][0].src, 8);
  ASSERT_EQ(blocks[0][0].dst, 33);
  ASSERT_EQ(blocks[0][1].src, 9);
  ASSERT_EQ(blocks[0][1].dst, 33);
  ASSERT_EQ(blocks[0][2].src, 13);
  ASSERT_EQ(blocks[0][2].dst, 33);
  ASSERT_EQ(blocks[0][3].src, 14);
  ASSERT_EQ(blocks[0][3].dst, 33);
  ASSERT_EQ(blocks[0][4].src, 15);
  ASSERT_EQ(blocks[0][4].dst, 33);
  ASSERT_EQ(blocks[0][5].src, 18);
  ASSERT_EQ(blocks[0][5].dst, 33);
  ASSERT_EQ(blocks[0][6].src, 19);
  ASSERT_EQ(blocks[0][6].dst, 33);
  ASSERT_EQ(blocks[0][7].src, 20);
  ASSERT_EQ(blocks[0][7].dst, 33);
  ASSERT_EQ(blocks[0][8].src, 22);
  ASSERT_EQ(blocks[0][8].dst, 33);
  ASSERT_EQ(blocks[0][9].src, 23);
  ASSERT_EQ(blocks[0][9].dst, 33);
  ASSERT_EQ(blocks[0][10].src, 26);
  ASSERT_EQ(blocks[0][10].dst, 33);
  ASSERT_EQ(blocks[0][11].src, 27);
  ASSERT_EQ(blocks[0][11].dst, 33);
  ASSERT_EQ(blocks[0][12].src, 28);
  ASSERT_EQ(blocks[0][12].dst, 33);
  ASSERT_EQ(blocks[0][13].src, 29);
  ASSERT_EQ(blocks[0][13].dst, 33);
  ASSERT_EQ(blocks[0][14].src, 30);
  ASSERT_EQ(blocks[0][14].dst, 33);
  ASSERT_EQ(blocks[0][15].src, 31);
  ASSERT_EQ(blocks[0][15].dst, 33);
  ASSERT_EQ(blocks[0][16].src, 32);
  ASSERT_EQ(blocks[0][16].dst, 33);
}

TEST_F(ServTest, KARATE_CLUB_2) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,4,4,5,5,5,6},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{1,2,3,4,5,6,7,8,10,11,12,13,2,3,7,13,3,7,8,9,13,7,12,13,6,10,6,10,16,16},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,1,1,1,1,2,2,2,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{17,19,21,31,17,19,21,30,27,28,32,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33},64));
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 2,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge0_src, edge0_dst)),
      .fanouts = NULL,
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 2,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge1_src, edge1_dst)),
      .fanouts = NULL,
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
  }
  std::vector<uint64_t> seeds{33};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 2);
  ASSERT_EQ(blocks[1].size(), 32);
  ASSERT_EQ(blocks[1][0].src, 0);
  ASSERT_EQ(blocks[1][0].dst, 8);
  ASSERT_EQ(blocks[1][1].src, 0);
  ASSERT_EQ(blocks[1][1].dst, 13);
  ASSERT_EQ(blocks[1][2].src, 0);
  ASSERT_EQ(blocks[1][2].dst, 19);
  ASSERT_EQ(blocks[1][3].src, 0);
  ASSERT_EQ(blocks[1][3].dst, 31);
  ASSERT_EQ(blocks[1][4].src, 1);
  ASSERT_EQ(blocks[1][4].dst, 13);
  ASSERT_EQ(blocks[1][5].src, 1);
  ASSERT_EQ(blocks[1][5].dst, 19);
  ASSERT_EQ(blocks[1][6].src, 1);
  ASSERT_EQ(blocks[1][6].dst, 30);
  ASSERT_EQ(blocks[1][7].src, 2);
  ASSERT_EQ(blocks[1][7].dst, 8);
  ASSERT_EQ(blocks[1][8].src, 2);
  ASSERT_EQ(blocks[1][8].dst, 9);
  ASSERT_EQ(blocks[1][9].src, 2);
  ASSERT_EQ(blocks[1][9].dst, 13);
  ASSERT_EQ(blocks[1][10].src, 2);
  ASSERT_EQ(blocks[1][10].dst, 27);
  ASSERT_EQ(blocks[1][11].src, 2);
  ASSERT_EQ(blocks[1][11].dst, 28);
  ASSERT_EQ(blocks[1][12].src, 2);
  ASSERT_EQ(blocks[1][12].dst, 32);
  ASSERT_EQ(blocks[1][13].src, 3);
  ASSERT_EQ(blocks[1][13].dst, 13);
  ASSERT_EQ(blocks[1][14].src, 8);
  ASSERT_EQ(blocks[1][14].dst, 30);
  ASSERT_EQ(blocks[1][15].src, 8);
  ASSERT_EQ(blocks[1][15].dst, 32);
  ASSERT_EQ(blocks[1][16].src, 14);
  ASSERT_EQ(blocks[1][16].dst, 32);
  ASSERT_EQ(blocks[1][17].src, 15);
  ASSERT_EQ(blocks[1][17].dst, 32);
  ASSERT_EQ(blocks[1][18].src, 18);
  ASSERT_EQ(blocks[1][18].dst, 32);
  ASSERT_EQ(blocks[1][19].src, 20);
  ASSERT_EQ(blocks[1][19].dst, 32);
  ASSERT_EQ(blocks[1][20].src, 22);
  ASSERT_EQ(blocks[1][20].dst, 32);
  ASSERT_EQ(blocks[1][21].src, 23);
  ASSERT_EQ(blocks[1][21].dst, 27);
  ASSERT_EQ(blocks[1][22].src, 23);
  ASSERT_EQ(blocks[1][22].dst, 29);
  ASSERT_EQ(blocks[1][23].src, 23);
  ASSERT_EQ(blocks[1][23].dst, 32);
  ASSERT_EQ(blocks[1][24].src, 24);
  ASSERT_EQ(blocks[1][24].dst, 27);
  ASSERT_EQ(blocks[1][25].src, 24);
  ASSERT_EQ(blocks[1][25].dst, 31);
  ASSERT_EQ(blocks[1][26].src, 25);
  ASSERT_EQ(blocks[1][26].dst, 31);
  ASSERT_EQ(blocks[1][27].src, 26);
  ASSERT_EQ(blocks[1][27].dst, 29);
  ASSERT_EQ(blocks[1][28].src, 28);
  ASSERT_EQ(blocks[1][28].dst, 31);
  ASSERT_EQ(blocks[1][29].src, 29);
  ASSERT_EQ(blocks[1][29].dst, 32);
  ASSERT_EQ(blocks[1][30].src, 30);
  ASSERT_EQ(blocks[1][30].dst, 32);
  ASSERT_EQ(blocks[1][31].src, 31);
  ASSERT_EQ(blocks[1][31].dst, 32);
}

TEST_F(ServTest, KARATE_CLUB_3) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,4,4,5,5,5,6},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{1,2,3,4,5,6,7,8,10,11,12,13,2,3,7,13,3,7,8,9,13,7,12,13,6,10,6,10,16,16},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,1,1,1,1,2,2,2,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{17,19,21,31,17,19,21,30,27,28,32,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33},64));
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 3,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge0_src, edge0_dst)),
      .fanouts = NULL,
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 3,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge1_src, edge1_dst)),
      .fanouts = NULL,
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
  }
  std::vector<uint64_t> seeds{33};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 3);
  ASSERT_EQ(blocks[2].size(), 19);
  ASSERT_EQ(blocks[2][0].src, 0);
  ASSERT_EQ(blocks[2][0].dst, 1);
  ASSERT_EQ(blocks[2][1].src, 0);
  ASSERT_EQ(blocks[2][1].dst, 2);
  ASSERT_EQ(blocks[2][2].src, 0);
  ASSERT_EQ(blocks[2][2].dst, 3);
  ASSERT_EQ(blocks[2][3].src, 0);
  ASSERT_EQ(blocks[2][3].dst, 8);
  ASSERT_EQ(blocks[2][4].src, 0);
  ASSERT_EQ(blocks[2][4].dst, 31);
  ASSERT_EQ(blocks[2][5].src, 1);
  ASSERT_EQ(blocks[2][5].dst, 2);
  ASSERT_EQ(blocks[2][6].src, 1);
  ASSERT_EQ(blocks[2][6].dst, 3);
  ASSERT_EQ(blocks[2][7].src, 1);
  ASSERT_EQ(blocks[2][7].dst, 30);
  ASSERT_EQ(blocks[2][8].src, 2);
  ASSERT_EQ(blocks[2][8].dst, 3);
  ASSERT_EQ(blocks[2][9].src, 2);
  ASSERT_EQ(blocks[2][9].dst, 8);
  ASSERT_EQ(blocks[2][10].src, 2);
  ASSERT_EQ(blocks[2][10].dst, 28);
  ASSERT_EQ(blocks[2][11].src, 8);
  ASSERT_EQ(blocks[2][11].dst, 30);
  ASSERT_EQ(blocks[2][12].src, 23);
  ASSERT_EQ(blocks[2][12].dst, 25);
  ASSERT_EQ(blocks[2][13].src, 23);
  ASSERT_EQ(blocks[2][13].dst, 29);
  ASSERT_EQ(blocks[2][14].src, 24);
  ASSERT_EQ(blocks[2][14].dst, 25);
  ASSERT_EQ(blocks[2][15].src, 24);
  ASSERT_EQ(blocks[2][15].dst, 31);
  ASSERT_EQ(blocks[2][16].src, 25);
  ASSERT_EQ(blocks[2][16].dst, 31);
  ASSERT_EQ(blocks[2][17].src, 26);
  ASSERT_EQ(blocks[2][17].dst, 29);
  ASSERT_EQ(blocks[2][18].src, 28);
  ASSERT_EQ(blocks[2][18].dst, 31);
}

TEST_F(ServTest, KARATE_CLUB_4) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,4,4,5,5,5,6},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{1,2,3,4,5,6,7,8,10,11,12,13,2,3,7,13,3,7,8,9,13,7,12,13,6,10,6,10,16,16},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,1,1,1,1,2,2,2,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{17,19,21,31,17,19,21,30,27,28,32,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33},64));
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 4,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge0_src, edge0_dst)),
      .fanouts = NULL,
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 4,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge1_src, edge1_dst)),
      .fanouts = NULL,
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
  }
  std::vector<uint64_t> seeds{33};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 4);
  ASSERT_EQ(blocks[3].size(), 8);
  ASSERT_EQ(blocks[3][0].src, 0);
  ASSERT_EQ(blocks[3][0].dst, 1);
  ASSERT_EQ(blocks[3][1].src, 0);
  ASSERT_EQ(blocks[3][1].dst, 2);
  ASSERT_EQ(blocks[3][2].src, 0);
  ASSERT_EQ(blocks[3][2].dst, 8);
  ASSERT_EQ(blocks[3][3].src, 1);
  ASSERT_EQ(blocks[3][3].dst, 2);
  ASSERT_EQ(blocks[3][4].src, 2);
  ASSERT_EQ(blocks[3][4].dst, 8);
  ASSERT_EQ(blocks[3][5].src, 2);
  ASSERT_EQ(blocks[3][5].dst, 28);
  ASSERT_EQ(blocks[3][6].src, 23);
  ASSERT_EQ(blocks[3][6].dst, 25);
  ASSERT_EQ(blocks[3][7].src, 24);
  ASSERT_EQ(blocks[3][7].dst, 25);
}

TEST_F(ServTest, KARATE_CLUB_5) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,4,4,5,5,5,6},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{1,2,3,4,5,6,7,8,10,11,12,13,2,3,7,13,3,7,8,9,13,7,12,13,6,10,6,10,16,16},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,1,1,1,1,2,2,2,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{17,19,21,31,17,19,21,30,27,28,32,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33},64));
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 5,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge0_src, edge0_dst)),
      .fanouts = NULL,
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 5,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge1_src, edge1_dst)),
      .fanouts = NULL,
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
  }
  std::vector<uint64_t> seeds{33};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 5);
  ASSERT_EQ(blocks[4].size(), 3);
  ASSERT_EQ(blocks[4][0].src, 0);
  ASSERT_EQ(blocks[4][0].dst, 1);
  ASSERT_EQ(blocks[4][1].src, 0);
  ASSERT_EQ(blocks[4][1].dst, 2);
  ASSERT_EQ(blocks[4][2].src, 1);
  ASSERT_EQ(blocks[4][2].dst, 2);
}

TEST_F(ServTest, FANOUT_TEST1) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{4,5},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{0,0},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{3,2,1},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{4,4,5},64));
    std::vector<int16_t> fanouts{1,1};
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 6,
      .num_layers = 2,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(6, edge0_src, edge0_dst)),
      .fanouts = fanouts.data(),
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 6,
      .num_layers = 2,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(6, edge1_src, edge1_dst)),
      .fanouts = fanouts.data(),
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
  }

  std::vector<uint64_t> seeds{0};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 2);
  ASSERT_EQ(blocks[0].size(), 1);
  ASSERT_EQ(blocks[1].size(), 1);
  ASSERT_EQ(blocks[0][0].src, 5);
  ASSERT_EQ(blocks[0][0].dst, 0);
  ASSERT_EQ(blocks[1][0].src, 1);
  ASSERT_EQ(blocks[1][0].dst, 5);
}


TEST_F(ServTest, FANOUT_KARATE_CLUB_1) {
  {
    dgl::IdArray edge0_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,4,4,5,5,5,6},64))
                ,edge0_dst(dgl::aten::VecToIdArray(std::vector<int>{1,2,3,4,5,6,7,8,10,11,12,13,2,3,7,13,3,7,8,9,13,7,12,13,6,10,6,10,16,16},64))
                ,edge1_src(dgl::aten::VecToIdArray(std::vector<int>{0,0,0,0,1,1,1,1,2,2,2,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32},64))
                ,edge1_dst(dgl::aten::VecToIdArray(std::vector<int>{17,19,21,31,17,19,21,30,27,28,32,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33},64));
    std::vector<int16_t> fanouts{5};
    neighbor_sampler_arg_t arg0 = {
      .rank = 0,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 1,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge0_src, edge0_dst)),
      .fanouts = fanouts.data(),
    };
    neighbor_sampler_arg_t arg1 = {
      .rank = 1,
      .size = 2,
      .num_nodes = 34,
      .num_layers = 1,
      .g = dgl::GraphRef(dgl::Graph::CreateFromCOO(34, edge1_src, edge1_dst)),
      .fanouts = fanouts.data(),
    };
    auto sampler0 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg0), &input0, &output0));
    auto sampler1 =
      std::unique_ptr<NeighborSampler>(new NeighborSampler(std::move(arg1), &input1, &output1));

    sm0.add_am_service(std::move(sampler0));
    sm1.add_am_service(std::move(sampler1));
  }
  std::vector<uint64_t> seeds{33};
  input0.push(std::move(seeds));
  while (output0.empty()) {
    sm0.progress();
    sm1.progress();
  }
  auto blocks = output0.front();
  ASSERT_EQ(blocks.size(), 1);
  ASSERT_EQ(blocks[0].size(), 5);
  ASSERT_EQ(blocks[0][0].src, 14);
  ASSERT_EQ(blocks[0][0].dst, 33);
  ASSERT_EQ(blocks[0][1].src, 20);
  ASSERT_EQ(blocks[0][1].dst, 33);
  ASSERT_EQ(blocks[0][2].src, 22);
  ASSERT_EQ(blocks[0][2].dst, 33);
  ASSERT_EQ(blocks[0][3].src, 23);
  ASSERT_EQ(blocks[0][3].dst, 33);
  ASSERT_EQ(blocks[0][4].src, 30);
  ASSERT_EQ(blocks[0][4].dst, 33);
}