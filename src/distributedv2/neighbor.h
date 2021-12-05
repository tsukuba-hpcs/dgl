/*!
 *  Copyright (c) 2021 by Contributors
 * \file distributedv2/neighbor.h
 * \brief headers for NeighborSampler
 */

#ifndef DGL_DISTV2_NEIGHBOR_H_
#define DGL_DISTV2_NEIGHBOR_H_

#include <stdlib.h>
#include <dgl/graph.h>
#include <memory>
#include <vector>
#include <atomic>
#include <queue>
#include <unordered_map>
#include <algorithm>


#include "service.h"

namespace dgl {
namespace distributedv2 {

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

using block_t = std::vector<edge_elem_t>;

struct neighbor_sampler_prog_t {
  std::vector<block_t> blocks;
  uint64_t ppt;
  neighbor_sampler_prog_t() : ppt(0) {}
  neighbor_sampler_prog_t(int num_layers)
  : blocks(num_layers)
  , ppt(0) {}
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
class NeighborSampler: public Service {
  GraphRef local_graph;
  uint16_t num_layers;
  std::vector<int16_t> fanouts;
  int rank, size;
  uint64_t node_slit;
  uint64_t req_id;
  static constexpr uint64_t PPT_ALL = 1000000000000ll;
  static constexpr size_t HEADER_LEN = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint16_t) + sizeof(uint16_t);
  std::queue<std::vector<dgl_id_t>> *input_que;
  std::queue<std::vector<block_t>>  *output_que;
  std::unordered_map<uint64_t, neighbor_sampler_prog_t> prog_que;
  void inline send_query(Communicator *comm, uint16_t dstrank, uint16_t depth, uint64_t req_id, dgl_id_t *nodes, uint16_t len, uint64_t ppt);
  void inline send_response(Communicator *comm, uint16_t depth, uint64_t req_id, edge_elem_t *edges, uint16_t len, uint64_t ppt);
  void inline recv_query(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint16_t len, const void *buffer);
  void inline recv_response(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint16_t len, const void *buffer);
  void scatter(Communicator *comm, uint16_t depth, uint64_t req_id, std::vector<dgl_id_t> &&seeds, uint64_t ppt);
public:
  NeighborSampler(neighbor_sampler_arg_t &&arg,
    std::queue<std::vector<dgl_id_t>> *input,
    std::queue<std::vector<block_t>> *output);
  void recv(Communicator *comm, const void *buffer, size_t length);
  void progress(Communicator *comm);
};


}
}

#endif