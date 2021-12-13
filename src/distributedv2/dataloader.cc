#include "dataloader.h"
#include "context.h"


#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include <numeric>
#include <random>

#include "../c_api_common.h"

namespace dgl {
namespace distributedv2 {

using namespace dgl::runtime;

NeighborSampler::NeighborSampler(neighbor_sampler_arg_t &&arg,
  ConcurrentQueue<seed_with_label_t> *input_que,
  std::queue<seed_with_blocks_t> *output_que)
  : rank(arg.rank)
  , size(arg.size)
  , node_slit((arg.num_nodes + arg.size - 1) / arg.size)
  , local_graph(arg.local_graph)
  , num_layers(arg.num_layers)
  , req_id(arg.rank)
  , input_que(input_que)
  , output_que(output_que)
  , prog_que({}) {
  fanouts.assign(num_layers, -1);
  if (!arg.fanouts.empty()) {
    CHECK(arg.fanouts.size() == num_layers);
    fanouts = arg.fanouts;
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
            prog_que[req_id].blocks[depth].edges.push_back(edge_elem_t{src,dst,id});
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
            prog_que[req_id].blocks[depth].edges.push_back(edge_elem_t{src,dst,id});
            next_seeds.push_back(src);
          }
        }
      }
      // LOG(INFO) << "next_seeds.size()" << next_seeds.size();
      if (next_seeds.size() == 0 || depth + 1 == num_layers) {
        prog_que[req_id].ppt += cur_ppt;
        // LOG(INFO) << "req_id=" << req_id << " ppt=" << ppt;
        if (prog_que[req_id].ppt == PPT_ALL) {
          enqueue(req_id);
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

void inline NeighborSampler::enqueue(uint64_t req_id) {
  LOG(INFO) << "req_id=" << req_id << " is finished";
  std::vector<uint64_t> src_nodes(prog_que[req_id].inputs.seeds);
  for (uint16_t dep = 0; dep < num_layers; dep++) {
    auto target_block = &prog_que[req_id].blocks[dep];
    std::sort(target_block->edges.begin(), target_block->edges.end());
    target_block->edges.erase(
      std::unique(target_block->edges.begin(), target_block->edges.end())
    , target_block->edges.end());
    for (edge_elem_t edge: target_block->edges) {
      src_nodes.push_back(edge.src);
    }
    std::sort(src_nodes.begin(), src_nodes.end());
    src_nodes.erase(
      std::unique(src_nodes.begin(), src_nodes.end())
    , src_nodes.end()
    );
    target_block->src_nodes = src_nodes;
  }
  output_que->push(seed_with_blocks_t(std::move(prog_que[req_id])));
  prog_que.erase(req_id);
}

void inline NeighborSampler::recv_response(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint16_t len, const void *buffer) {
  edges_t edges(len);
  std::memcpy(edges.data(), buffer, sizeof(edge_elem_t) * len);
  block_t *target_block = &prog_que[req_id].blocks[depth];
  target_block->edges.insert(target_block->edges.end(), edges.begin(), edges.end());
  prog_que[req_id].ppt += ppt;
  // LOG(INFO) << "req_id=" << req_id << " ppt=" << ppt;
  if (prog_que[req_id].ppt == PPT_ALL) {
    enqueue(req_id);
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
  seed_with_label_t input;
  while (input_que->try_dequeue(input)) {
    prog_que[req_id] = neighbor_sampler_prog_t(num_layers, std::move(input));
    scatter(comm, 0, req_id, std::vector<dgl_id_t>(prog_que[req_id].inputs.seeds), PPT_ALL);
    req_id += size;
  }
}

FeatLoader::FeatLoader(feat_loader_arg_t &&arg,
  std::queue<seed_with_blocks_t>  *input_que,
  ConcurrentQueue<seed_with_feat_t> *output_que)
: rank(arg.rank)
, size(arg.size)
, node_slit((arg.num_nodes + arg.size - 1) / arg.size)
, req_id(arg.rank)
, local_feats(std::move(arg.local_feats))
, input_que(input_que)
, output_que(output_que) {
  int64_t row = 1;
  for (int dim = 1; dim < local_feats->ndim; dim++) {
    row *= local_feats->shape[dim];
  }
  feats_row = row;
  feats_row_size = feats_row * local_feats->dtype.bits / 8;
}

std::pair<void *, size_t> FeatLoader::served_buffer() {
  int64_t length = local_feats->dtype.bits / 8;
  for (int dim = 0; dim < local_feats->ndim; dim++) {
    length *= local_feats->shape[dim];
  }
  return std::make_pair(local_feats->data, length);
}

void FeatLoader::progress(Communicator *comm) {
  while (!input_que->empty()) {
    CHECK(input_que->front().blocks.size() > 0);
    seed_with_blocks_t item = std::move(input_que->front());
    CHECK(item.blocks.size() > 0);
    prog_que[req_id] = feat_loader_prog_t(std::move(item));
    input_que->pop();
    std::vector<dgl_id_t> &input_nodes = prog_que[req_id].inputs.blocks.back().src_nodes;
    std::vector<int64_t> shape{
      static_cast<int64_t>(input_nodes.size()),
      static_cast<int64_t>(feats_row)
    };
    prog_que[req_id].feats = NDArray::Empty(shape, local_feats->dtype, DLContext{kDLCPU, 0});

    for (size_t row = 0; row < shape[0]; row++) {
      dgl_id_t node = input_nodes[row];
      int src_rank = node / node_slit;
      uint64_t offset = (node % node_slit) * feats_row_size;
      void *recv_buffer = PTR_BYTE_OFFSET(prog_que[req_id].feats->data, feats_row_size * row);
      if (src_rank == rank) {
        std::memcpy(recv_buffer, PTR_BYTE_OFFSET(served_buffer().first, offset), feats_row_size);
        prog_que[req_id].received++;
        if (prog_que[req_id].received == prog_que[req_id].num_input_nodes) {
          LOG(INFO) << "req_id=" << req_id << " is completed";
          output_que->enqueue(seed_with_feat_t(std::move(prog_que[req_id])));
          prog_que.erase(req_id);
        }
        continue;
      }
      comm->rma_read(src_rank, rma_id, req_id, recv_buffer, offset, feats_row_size);
    }
    req_id += size;
  }
}

void FeatLoader::rma_read_cb(Communicator *comm, uint64_t req_id, void *buffer) {
  prog_que[req_id].received++;
  if (prog_que[req_id].received == prog_que[req_id].num_input_nodes) {
    LOG(INFO) << "rma_read_cb: req_id=" << req_id << " is completed";
    output_que->enqueue(seed_with_feat_t(std::move(prog_que[req_id])));
    prog_que.erase(req_id);
  }
}


NodeDataLoader::NodeDataLoader(Communicator *comm, node_dataloader_arg_t &&arg)
: ServiceManager(arg.rank, arg.size, comm) {
  
}

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "_CAPI_DistV2CreateNodeDataLoader is called";
  ContextRef ctx = args[0];
  int num_layers = args[1];
  int num_nodes = args[2];
  GraphRef local_graph = args[3];
  List<Value> _fanouts = args[4];
  std::vector<int> fanouts(ListValueToVector<int>(_fanouts));
  NDArray local_feats = args[5];
  node_dataloader_arg_t arg = {
    .rank = ctx->rank,
    .size = ctx->size,
    .num_nodes = num_nodes,
    .num_layers = num_layers,
    .local_graph = std::move(local_graph),
    .fanouts = fanouts,
    .local_feats = std::move(local_feats),
  };
  std::shared_ptr<NodeDataLoader> loader(new NodeDataLoader(&ctx->comm, std::move(arg)));
  *rv = loader;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2EnqueueToNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  List<Value> _seeds = args[1];
  std::vector<dgl_id_t> seeds(ListValueToVector<dgl_id_t>(_seeds));
  NDArray labels = args[2];
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2DequeueToNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  *rv = 0;
});

}
}