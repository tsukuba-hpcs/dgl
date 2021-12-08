#include "neighbor.h"
#include <numeric>
#include <random>

namespace dgl {
namespace distributedv2 {


NeighborSampler::NeighborSampler(neighbor_sampler_arg_t &&arg,
  std::queue<std::vector<dgl_id_t>> *input,
  std::queue<std::vector<block_t>> *output)
  : rank(arg.rank)
  , size(arg.size)
  , node_slit((arg.num_nodes + arg.size - 1) / arg.size)
  , local_graph(arg.g)
  , num_layers(arg.num_layers)
  , req_id(arg.rank)
  , input_que(input)
  , output_que(output)
  , prog_que({}) {
  fanouts.assign(num_layers, -1);
  if (arg.fanouts) {
    std::memcpy(fanouts.data(), arg.fanouts, num_layers * sizeof(int16_t));
  }
}

void inline NeighborSampler::send_query(Communicator *comm, uint16_t dstrank, uint16_t depth, uint64_t req_id, uint64_t *nodes, uint16_t len, uint64_t ppt) {
  size_t data_len = HEADER_LEN + len * sizeof(dgl_id_t);
  size_t offset = 0;
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[data_len]);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = (req_id<<1);
  offset += sizeof(uint64_t);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = ppt;
  offset += sizeof(uint64_t);
  *(uint16_t *)PTR_BYTE_OFFSET(data.get(), offset) = depth;
  offset += sizeof(uint16_t);
  *(uint16_t *)PTR_BYTE_OFFSET(data.get(), offset) = len;
  offset += sizeof(uint16_t);
  std::memcpy(PTR_BYTE_OFFSET(data.get(), offset), nodes, len * sizeof(dgl_id_t));
  comm->am_post(dstrank, am_id, std::move(data), data_len);
}

void inline NeighborSampler::send_response(Communicator *comm, uint16_t depth, uint64_t req_id, edge_elem_t *edges, uint16_t len, uint64_t ppt) {
  uint16_t dstrank = req_id % size;
  size_t data_len = HEADER_LEN + len * sizeof(edge_elem_t);
  size_t offset = 0;
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[data_len]);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = (req_id<<1) + 1;
  offset += sizeof(uint64_t);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = ppt;
  offset += sizeof(uint64_t);
  *(uint16_t *)PTR_BYTE_OFFSET(data.get(), offset) = depth;
  offset += sizeof(uint16_t);
  *(uint16_t *)PTR_BYTE_OFFSET(data.get(), offset) = len;
  offset += sizeof(uint16_t);
  std::memcpy(PTR_BYTE_OFFSET(data.get(), offset), edges, len * sizeof(edge_elem_t));
  comm->am_post(dstrank, am_id, std::move(data), data_len);
}

void NeighborSampler::scatter(Communicator *comm, uint16_t depth, uint64_t req_id, std::vector<dgl_id_t> &&seeds, uint64_t ppt) {
  CHECK(ppt > 0);
  uint64_t rem_ppt = ppt;
  std::sort(seeds.begin(), seeds.end());
  std::minstd_rand0 engine(req_id ^ ppt ^ depth);
  for (uint16_t dstrank = 0, l, r = 0; dstrank < size; dstrank++) {
    l = r;
    while (r < seeds.size() && seeds[r] / node_slit == dstrank) {
      r++;
    }
    if (l == r) continue;
    // LOG(INFO) << "scatter depth=" << depth <<  " rank=" << rank << " dstrank=" << dstrank << "req_id" << req_id;
    for (uint16_t i = l; i < r; i++) {
      // LOG(INFO) << "i=" << i << " seeds[i]=" << seeds[i];
    }
    // Relay the query
    if (dstrank != rank) {
      uint64_t cur_ppt = (r == seeds.size()) ? rem_ppt : ppt * (r-l) / seeds.size();
      CHECK(cur_ppt > 0);
      rem_ppt -= cur_ppt;
      send_query(comm, dstrank, depth, req_id, &seeds[l], r-l, cur_ppt);
      continue;
    }
    // Handle query
    uint64_t cur_ppt = (r == seeds.size()) ? rem_ppt : ppt * (r-l) / seeds.size();
    CHECK(cur_ppt > 0);
    rem_ppt -= cur_ppt;
    // self request
    if (req_id % size == rank) {
      dgl_id_t src, dst, id;
      std::vector<dgl_id_t> next_seeds;
      for (uint16_t dst_idx = l; dst_idx < r; dst_idx++) {
        dgl::EdgeArray src_edges = local_graph->InEdges(aten::VecToIdArray(std::vector<dgl_id_t>{seeds[dst_idx]}, 64));
        int64_t edge_len = src_edges.id.NumElements();
        // LOG(INFO) << "dst_idx= " << dst_idx << " edge_len=" << edge_len << "depth=" << depth << "fanouts[depth]= " << fanouts[depth];
        if (fanouts[depth] < 0 || edge_len <= fanouts[depth]) {
          for (uint16_t idx = 0; idx < edge_len; idx++) {
            src = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.src->data, sizeof(dgl_id_t) * idx);
            dst = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.dst->data, sizeof(dgl_id_t) * idx);
            id = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.id->data, sizeof(dgl_id_t) * idx);
            // LOG(INFO) << "self all: idx=" << idx << " src=" << src << " dst=" << dst;
            prog_que[req_id].blocks[depth].push_back(edge_elem_t{src,dst,id});
            next_seeds.push_back(src);
          }
        } else {
          // sampling
          std::vector<uint16_t> seq(edge_len);
          std::iota(seq.begin(), seq.end(), 0);
          for (uint16_t idx = edge_len-1; idx >= fanouts[depth]; idx--) {
            std::swap(seq[idx], seq[engine() % idx]);
          }
          for (uint16_t idx = 0; idx < fanouts[depth]; idx++) {
            src = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.src->data, sizeof(dgl_id_t) * seq[idx]);
            dst = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.dst->data, sizeof(dgl_id_t) * seq[idx]);
            id = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.id->data, sizeof(dgl_id_t) * seq[idx]);
            // LOG(INFO) << "self sample: idx=" << idx << " src=" << src << " dst=" << dst;
            prog_que[req_id].blocks[depth].push_back(edge_elem_t{src,dst,id});
            next_seeds.push_back(src);
          }
        }
      }
      // LOG(INFO) << "next_seeds.size()" << next_seeds.size();
      if (next_seeds.size() == 0 || depth + 1 == num_layers) {
        prog_que[req_id].ppt += cur_ppt;
        // LOG(INFO) << "req_id=" << req_id << " ppt=" << ppt;
        if (prog_que[req_id].ppt == PPT_ALL) {
          // LOG(INFO) << "req_id=" << req_id << " is finished";
          for (uint16_t dep = 0; dep < num_layers; dep++) {
            std::sort(prog_que[req_id].blocks[dep].begin(), prog_que[req_id].blocks[dep].end());
            prog_que[req_id].blocks[dep].erase(
              std::unique(prog_que[req_id].blocks[dep].begin(), prog_que[req_id].blocks[dep].end())
            , prog_que[req_id].blocks[dep].end());
          }
          output_que->push(std::move(prog_que[req_id].blocks));
          prog_que.erase(req_id);
        }
      } else {
        scatter(comm, depth + 1, req_id, std::move(next_seeds), cur_ppt);
      }
    // other request
    } else {
      std::vector<edge_elem_t> edges;
      std::vector<dgl_id_t> next_seeds;
      for (uint16_t dst_idx = l; dst_idx < r; dst_idx++) {
        dgl::EdgeArray src_edges = local_graph->InEdges(aten::VecToIdArray(std::vector<dgl_id_t>{seeds[dst_idx]}, 64));
        int64_t edge_len = src_edges.id.NumElements();
        // LOG(INFO) << "dst_idx= " << dst_idx << " edge_len=" << edge_len << "depth=" << depth << "fanouts[depth]= " << fanouts[depth];
        if (fanouts[depth] < 0 || edge_len <= fanouts[depth]) {
          for (uint16_t idx = 0; idx < edge_len; idx++) {
            edge_elem_t elem;
            elem.src = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.src->data, sizeof(dgl_id_t) * idx);
            elem.dst = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.dst->data, sizeof(dgl_id_t) * idx);
            elem.id = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.id->data, sizeof(dgl_id_t) * idx);
            // LOG(INFO) << "other all: idx=" << idx << " src=" << elem.src << " dst=" << elem.dst;
            next_seeds.push_back(elem.src);
            edges.push_back(std::move(elem));
          }
        } else {
          // sampling
          std::vector<uint16_t> seq(edge_len);
          std::iota(seq.begin(), seq.end(), 0);
          for (uint16_t idx = edge_len-1; idx >= fanouts[depth]; idx--) {
            std::swap(seq[idx], seq[engine() % idx]);
          }
          for (uint16_t idx = 0; idx < fanouts[depth]; idx++) {
            edge_elem_t elem;
            elem.src = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.src->data, sizeof(dgl_id_t) * seq[idx]);
            elem.dst = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.dst->data, sizeof(dgl_id_t) * seq[idx]);
            elem.id = *(dgl_id_t *)PTR_BYTE_OFFSET(src_edges.id->data, sizeof(dgl_id_t) * seq[idx]);
            // LOG(INFO) << "other sample: idx=" << idx << " src=" << elem.src << " dst=" << elem.dst;
            next_seeds.push_back(elem.src);
            edges.push_back(std::move(elem));
          }
        }
      }
      if (next_seeds.size() == 0 || depth + 1 == num_layers) {
        send_response(comm, depth, req_id, edges.data(), edges.size(), cur_ppt);
      } else {
        send_response(comm, depth, req_id, edges.data(), edges.size(), 1);
        scatter(comm, depth + 1, req_id, std::move(next_seeds), cur_ppt - 1);
      }
    }
  }
}

void inline NeighborSampler::recv_query(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint16_t len, const void *buffer) {
  std::vector<dgl_id_t> seeds(len);
  std::memcpy(seeds.data(), buffer, sizeof(dgl_id_t) * len);
  scatter(comm, depth, req_id, std::move(seeds), ppt);
}

void inline NeighborSampler::recv_response(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint16_t len, const void *buffer) {
  block_t edges(len);
  std::memcpy(edges.data(), buffer, sizeof(edge_elem_t) * len);
  prog_que[req_id].blocks[depth].insert(prog_que[req_id].blocks[depth].end(), edges.begin(), edges.end());
  prog_que[req_id].ppt += ppt;
  // LOG(INFO) << "req_id=" << req_id << " ppt=" << ppt;
  if (prog_que[req_id].ppt == PPT_ALL) {
    // LOG(INFO) << "req_id=" << req_id << " is finished";
    for (uint16_t dep = 0; dep < num_layers; dep++) {
      std::sort(prog_que[req_id].blocks[dep].begin(), prog_que[req_id].blocks[dep].end());
      prog_que[req_id].blocks[dep].erase(
        std::unique(prog_que[req_id].blocks[dep].begin(), prog_que[req_id].blocks[dep].end())
      , prog_que[req_id].blocks[dep].end());
    }
    output_que->push(std::move(prog_que[req_id].blocks));
    prog_que.erase(req_id);
  }
}

void NeighborSampler::am_recv(Communicator *comm, const void *buffer, size_t length) {
  size_t offset = 0;
  uint64_t shifted_id;
  uint16_t depth, data_length;
  uint64_t ppt;
  shifted_id = *(uint64_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint64_t);
  ppt = *(uint64_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint64_t);
  depth = *(uint16_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint16_t);
  data_length = *(uint16_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint16_t);

  if (shifted_id & 1) {
    recv_response(comm, depth, ppt, shifted_id>>1, data_length, PTR_BYTE_OFFSET(buffer, offset));
    CHECK(offset + sizeof(edge_elem_t) * data_length == length);
  } else {
    recv_query(comm, depth, ppt, shifted_id>>1, data_length, PTR_BYTE_OFFSET(buffer, offset));
    CHECK(offset + sizeof(dgl_id_t) * data_length == length);
  }
}

void NeighborSampler::progress(Communicator *comm) {
  while (!input_que->empty()) {
    std::vector<dgl_id_t> seeds = input_que->front();
    input_que->pop();
    prog_que[req_id] = neighbor_sampler_prog_t(num_layers);
    scatter(comm, 0, req_id, std::move(seeds), PPT_ALL);
    req_id += size;
  }
}

}
}