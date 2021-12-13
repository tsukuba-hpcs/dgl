/*!
 *  Copyright (c) 2021 by Contributors
 * \file distributedv2/dataloader.h
 * \brief headers for distv2 dataloader.
 */

#ifndef DGL_DISTV2_DATALOADER_H_
#define DGL_DISTV2_DATALOADER_H_

#include "service.h"
#include <dmlc/concurrentqueue.h>
#include <vector>
#include <queue>

namespace dgl {
namespace distributedv2 {

using namespace dmlc::moodycamel;

struct seed_with_label_t {
  std::vector<dgl_id_t> seeds;
  NDArray labels;
};


struct edge_elem_t {
  dgl_id_t src, dst, id;
  bool operator==(const edge_elem_t& rhs) const {
    return src == rhs.src && dst == rhs.dst;
  }
  bool operator<(const edge_elem_t& rhs) const {
    if (src != rhs.src) return src < rhs.src;
    return dst < rhs.dst;
  }
};


using edges_t = std::vector<edge_elem_t>;

struct block_t {
  edges_t edges;
  std::vector<dgl_id_t> src_nodes;
};

// Neighbor Sampler

struct neighbor_sampler_prog_t {
  seed_with_label_t inputs;
  std::vector<block_t> blocks;
  uint64_t ppt;
  neighbor_sampler_prog_t() : ppt(0) {}
  neighbor_sampler_prog_t(int num_layers, seed_with_label_t &&inputs)
  : inputs(std::move(inputs))
  , blocks(num_layers)
  , ppt(0) {}
};

struct seed_with_blocks_t {
  std::vector<dgl_id_t> seeds;
  NDArray labels;
  std::vector<block_t> blocks;
  seed_with_blocks_t() {}
  seed_with_blocks_t(neighbor_sampler_prog_t &&prog)
  : seeds(std::move(prog.inputs.seeds))
  , labels(std::move(prog.inputs.labels))
  , blocks(std::move(prog.blocks)) {}
  seed_with_blocks_t(std::vector<dgl_id_t> &&seeds, NDArray &&labels, std::vector<block_t> &&blocks)
  : seeds(std::move(seeds))
  , labels(std::move(labels))
  , blocks(std::move(blocks)) {}
};

struct neighbor_sampler_arg_t {
  int rank;
  int size;
  uint64_t num_nodes;
  uint16_t num_layers;
  GraphRef g;
  int16_t *fanouts;
};

/**
 * Binary format of the request:
 * uint64_t (req_id<<1) | uint64_t ppt | uint16_t depth | uint16_t nodes' length | [uint64_t node_id] * length
 * Binary format of the response:
 * uint64_t (req_id<<1)+1 | uint64_t ppt | uint16_t depth | uint16_t edges' length | [uint64_t src_id uint64_t dst_id uint64_t edge_id] * length
 */
class NeighborSampler: public AMService {
  GraphRef local_graph;
  uint16_t num_layers;
  std::vector<int16_t> fanouts;
  int rank, size;
  uint64_t node_slit;
  uint64_t req_id;
  static constexpr uint64_t PPT_ALL = 1000000000000ll;
  static constexpr size_t HEADER_LEN = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint16_t) + sizeof(uint16_t);
  std::queue<seed_with_label_t> *input_que;
  std::queue<seed_with_blocks_t>  *output_que;
  std::unordered_map<uint64_t, neighbor_sampler_prog_t> prog_que;
  void inline enqueue(uint64_t req_id);
  void inline send_query(Communicator *comm, uint16_t dstrank, uint16_t depth, uint64_t req_id, dgl_id_t *nodes, uint16_t len, uint64_t ppt);
  void inline send_response(Communicator *comm, uint16_t depth, uint64_t req_id, edge_elem_t *edges, uint16_t len, uint64_t ppt);
  void inline recv_query(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint16_t len, const void *buffer);
  void inline recv_response(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint16_t len, const void *buffer);
  void scatter(Communicator *comm, uint16_t depth, uint64_t req_id, std::vector<dgl_id_t> &&seeds, uint64_t ppt);
public:
  NeighborSampler(neighbor_sampler_arg_t &&arg,
    std::queue<seed_with_label_t> *input,
    std::queue<seed_with_blocks_t> *output);
  void am_recv(Communicator *comm, const void *buffer, size_t length);
  void progress(Communicator *comm);
};

struct feat_loader_prog_t {
  seed_with_blocks_t inputs;
  NDArray feats;
  uint64_t received;
  uint64_t num_input_nodes;
  feat_loader_prog_t(): received(0) {}
  feat_loader_prog_t(seed_with_blocks_t &&_inputs)
  : inputs(std::move(_inputs))
  , received(0) {
    CHECK(inputs.blocks.size() > 0);
    num_input_nodes = inputs.blocks.back().src_nodes.size();
  }
};

struct seed_with_feat_t {
  std::vector<dgl_id_t> seeds;
  NDArray labels;
  std::vector<block_t> blocks;
  NDArray feats;
  seed_with_feat_t(feat_loader_prog_t &&prog)
  : seeds(std::move(prog.inputs.seeds))
  , labels(std::move(prog.inputs.labels))
  , blocks(std::move(prog.inputs.blocks))
  , feats(std::move(prog.feats)) {
  }
  seed_with_feat_t() {}
};

struct feat_loader_arg_t {
  int rank;
  int size;
  uint64_t num_nodes;
  NDArray local_feats;
};

class FeatLoader: public RMAService {
  NDArray local_feats;
  uint64_t feats_row;
  uint64_t feats_row_size;
  int rank, size;
  uint64_t node_slit;
  uint64_t req_id;
  std::queue<seed_with_blocks_t>  *input_que;
  ConcurrentQueue<seed_with_feat_t> *output_que;
  std::unordered_map<uint64_t, feat_loader_prog_t> prog_que;
public:
  FeatLoader(feat_loader_arg_t &&arg,
    std::queue<seed_with_blocks_t>  *input_que,
    ConcurrentQueue<seed_with_feat_t> *output_que
  );
  std::pair<void *, size_t> served_buffer();
  void rma_read_cb(Communicator *comm, uint64_t req_id, void *buffer);
  void progress(Communicator *comm);
};

struct node_dataloader_arg_t {
  uint64_t num_nodes;
  uint16_t num_layers;
};

class NodeDataLoader: public ServiceManager, public runtime::Object {
  ConcurrentQueue<seed_with_label_t> input;
  ConcurrentQueue<std::vector<int>> output;
public:
  static constexpr const char* _type_key = "distributedv2.NodeDataLoader";
  NodeDataLoader(int rank, int size, Communicator *comm, node_dataloader_arg_t &&arg);
  DGL_DECLARE_OBJECT_TYPE_INFO(NodeDataLoader, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(NodeDataLoaderRef, NodeDataLoader);

}
}

#endif