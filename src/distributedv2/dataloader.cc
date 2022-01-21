#include "dataloader.h"


#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include <numeric>
#include <random>
#include <chrono>
#include <algorithm>

#include "../c_api_common.h"

namespace dgl {
namespace distributedv2 {

using namespace dgl::runtime;

static int64_t dequeue_time = -1;
static int64_t build_block_time = 0;
static int64_t self_scatter_time = 0;
static int64_t handle_req_time = 0;
static int64_t recv_query_time = 0;
static int64_t create_idarray_time = 0;
static int64_t toblock_time = 0;
static int64_t create_graph_time = 0;
static int64_t edge_copy_time = 0;
static int64_t erase_time = 0;
static int64_t memcpy_time = 0;
static size_t batch_size_total = 0;
static size_t batch_num_total = 0;

edge_shard_t::edge_shard_t(NDArray &&_src, NDArray &&_dst, int rank, int size, uint64_t num_nodes)
: src(std::move(_src))
, dst(std::move(_dst))
, rank(rank)
, node_slit((num_nodes + size - 1) / size) {
  if (src->dtype.code != DLDataTypeCode::kDLInt || src->dtype.bits != 64) {
    LOG(FATAL) << "edge src dtype is invalid";
  }
  if (dst->dtype.code != DLDataTypeCode::kDLInt || dst->dtype.bits != 64) {
    LOG(FATAL) << "edge dst dtype is invalid";
  }
  if (src->ndim != dst->ndim || src.NumElements() != dst.NumElements()) {
    LOG(FATAL) << "edge src and dst must be same shape";
  }
  for (int64_t d = 0; d < src->ndim; d++) {
    CHECK(src->shape[d] == dst->shape[d]);
  }
  // Check dst id is between [node_slit * rank, node_slit * (rank+1))
  for (dgl_id_t idx = 0; idx < dst.NumElements(); idx++) {
    node_id_t dst_id = *(node_id_t *)PTR_BYTE_OFFSET(dst->data, sizeof(node_id_t) * idx);
    CHECK(node_slit * rank <= dst_id);
    CHECK(dst_id < node_slit * (rank+1));
  }
  // Check dst id is sorted
  for (dgl_id_t idx = 1; idx < dst.NumElements(); idx++) {
    node_id_t dst_id0 = *(node_id_t *)PTR_BYTE_OFFSET(dst->data, sizeof(node_id_t) * (idx-1));
    node_id_t dst_id1 = *(node_id_t *)PTR_BYTE_OFFSET(dst->data, sizeof(node_id_t) * idx);
    CHECK(dst_id0 <= dst_id1);
    node_id_t src_id0 = *(node_id_t *)PTR_BYTE_OFFSET(src->data, sizeof(node_id_t) * (idx-1));
    node_id_t src_id1 = *(node_id_t *)PTR_BYTE_OFFSET(src->data, sizeof(node_id_t) * idx);
    CHECK(dst_id0 < dst_id1 || src_id0 < src_id1);
  }
  offset.assign(node_slit + 1, src.NumElements());
  // edge where dst_id==k -> [offset[k], offset[k+1])
  for (dgl_id_t idx = 0; idx < src.NumElements(); idx++) {
    node_id_t dst_id = *(node_id_t *)PTR_BYTE_OFFSET(dst->data, sizeof(node_id_t) * idx);
    offset[dst_id % node_slit] = std::min(offset[dst_id % node_slit], idx);
  }
  for (dgl_id_t id = node_slit; id > 0; id--) {
    offset[id-1] = std::min(offset[id-1], offset[id]);
  }
}

void edge_shard_t::in_edges(node_id_t **src_ids, size_t *length, node_id_t dst_id) {
  CHECK(node_slit * rank <= dst_id);
  CHECK(dst_id < node_slit * (rank+1));
  CHECK(offset[dst_id % node_slit] <= offset[dst_id % node_slit + 1]);
  *length = offset[dst_id % node_slit + 1] - offset[dst_id % node_slit];
  if (*length == 0) {
    *src_ids = NULL;
    return;
  }
  *src_ids = (node_id_t *)PTR_BYTE_OFFSET(src->data, sizeof(node_id_t) * offset[dst_id % node_slit]);
}

NDArrayPool::NDArrayPool(size_t capacity)
: buffer(capacity)
, ready(true)
{}

void NDArrayPool::dump_chunks() {
  for (auto chunk: chunks) {
    LOG(INFO) << "chunk offset=" << chunk.offset << " length=" << chunk.length;
  }
}

NDArray NDArrayPool::alloc(std::vector<int64_t> shape, DLDataType dtype) {
  size_t length = dtype.bits / 8;
  for (uint16_t dim = 0; dim < shape.size(); dim++) {
    CHECK(shape[dim] > 0);
    length *= shape[dim];
  }
  size_t offset = 0;
  // sync
  {
    bool expected;
    do {
      expected = true;
      ready.compare_exchange_weak(expected, false, std::memory_order_acquire);
    } while (!expected);
  }
  if (!chunks.empty()) {
    offset = chunks.back().offset + chunks.back().length;
    if (offset + length > buffer.size()) {
      // [new chunk] -- [front] -- [back]
      offset = 0;
      if (length > chunks.front().offset) {
        dump_chunks();
        LOG(FATAL)
          << "NDArrayPool::alloc() failed with"
          << " chunks.front().offset=" << chunks.front().offset
          << " offset=" << offset
          << " length=" << length;
      }
    }
    // [back] -- [new chunk] -- [front]
    // check                 |<- this border
    if (chunks.back().offset < chunks.front().offset && offset + length > chunks.front().offset) {
      dump_chunks();
      LOG(FATAL)
        << "NDArrayPool::alloc() failed with"
        << " chunks.front().offset=" << chunks.front().offset
        << " offset=" << offset
        << " length=" << length;
    }
  }
  // [back] -- [new chunk] |
  // check                 |<- this border
  if (offset + length > buffer.size()) {
    dump_chunks();
    LOG(FATAL)
      << "NDArrayPool::alloc() failed with"
      << " buffer.size()=" << buffer.size()
      << " offset=" << offset
      << " length=" << length;
  }
  ndarray_pool_item_t item{
    .length = length,
    .offset = offset,
  };
  chunks.push_back(std::move(item));
  // sync
  {
    ready.store(true, std::memory_order_release);
  }
  DLTensor tensor;
  tensor.ctx = DLContext{kDLCPU, 0};
  tensor.ndim = static_cast<int>(shape.size());
  tensor.dtype = dtype;
  tensor.shape = new int64_t[tensor.ndim];
  for (int i = 0; i < tensor.ndim; ++i) {
    tensor.shape[i] = shape[i];
  }
  tensor.strides = new int64_t[tensor.ndim];
  for (int i = 0; i < tensor.ndim; ++i) {
    tensor.strides[i] = 1;
  }
  for (int i = tensor.ndim - 2; i >= 0; --i) {
    tensor.strides[i] = tensor.shape[i+1] * tensor.strides[i+1];
  }
  tensor.data = PTR_BYTE_OFFSET(&buffer[0], offset);
  DLManagedTensor *managed_tensor = new DLManagedTensor();
  managed_tensor->dl_tensor = std::move(tensor);
  managed_tensor->deleter = NDArrayPool::release;
  managed_tensor->manager_ctx = this;
  return NDArray::FromDLPack(managed_tensor);
}

void NDArrayPool::release(DLManagedTensor* managed_tensor) {
  NDArrayPool *self = (NDArrayPool *)managed_tensor->manager_ctx;
  size_t length = managed_tensor->dl_tensor.dtype.bits / 8;
  for (uint16_t dim = 0; dim < managed_tensor->dl_tensor.ndim; dim++) {
    CHECK(managed_tensor->dl_tensor.shape[dim] > 0);
    length *= managed_tensor->dl_tensor.shape[dim];
  }
  bool found = false;
  auto ptr = managed_tensor->dl_tensor.data;
  // sync
  {
    bool expected;
    do {
      expected = true;
      self->ready.compare_exchange_weak(expected, false, std::memory_order_acquire);
    } while (!expected);
  }
  for (auto itr = self->chunks.begin(); itr != self->chunks.end();) {
    if (PTR_BYTE_OFFSET(&self->buffer[0], itr->offset) == ptr) {
      CHECK(!found) << "offset is duplicated";
      found = true;
      itr = self->chunks.erase(itr);
      continue;
    }
    itr++;
  }
  // sync
  {
    self->ready.store(true, std::memory_order_release);
  }
  if (!found) {
    LOG(FATAL) << "NDArrayPool::release(): chunk not found";
  }
  delete [] managed_tensor->dl_tensor.shape;
  delete [] managed_tensor->dl_tensor.strides;
  delete managed_tensor;
}

NeighborSampler::NeighborSampler(neighbor_sampler_arg_t &&arg,
  BlockingConcurrentQueue<seed_with_label_t> *input_que,
  std::queue<blocks_with_label_t> *output_que)
  : rank(arg.rank)
  , size(arg.size)
  , node_slit((arg.num_nodes + arg.size - 1) / arg.size)
  , edge_shard(std::move(arg.edge_shard))
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

void inline NeighborSampler::send_query(Communicator *comm, uint16_t dstrank, uint16_t depth, uint64_t req_id, node_id_t *nodes, uint32_t len, uint64_t ppt) {
  size_t data_len = HEADER_LEN + len * sizeof(node_id_t);
  size_t offset = 0;
  auto mstart = std::chrono::system_clock::now();
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[data_len]);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = (req_id<<1);
  offset += sizeof(uint64_t);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = ppt;
  offset += sizeof(uint64_t);
  *(uint16_t *)PTR_BYTE_OFFSET(data.get(), offset) = depth;
  offset += sizeof(uint16_t);
  *(uint32_t *)PTR_BYTE_OFFSET(data.get(), offset) = len;
  offset += sizeof(uint32_t);
  std::memcpy(PTR_BYTE_OFFSET(data.get(), offset), nodes, len * sizeof(node_id_t));
  memcpy_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mstart).count();
  comm->am_post(dstrank, am_id, std::move(data), data_len);
}

void inline NeighborSampler::send_response(Communicator *comm, uint16_t depth, uint64_t req_id, edge_elem_t *edges, uint32_t len, uint64_t ppt) {
  uint16_t dstrank = req_id % size;
  size_t data_len = HEADER_LEN + len * sizeof(edge_elem_t);
  size_t offset = 0;
  auto mstart = std::chrono::system_clock::now();
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(new uint8_t[data_len]);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = (req_id<<1) + 1;
  offset += sizeof(uint64_t);
  *(uint64_t *)PTR_BYTE_OFFSET(data.get(), offset) = ppt;
  offset += sizeof(uint64_t);
  *(uint16_t *)PTR_BYTE_OFFSET(data.get(), offset) = depth;
  offset += sizeof(uint16_t);
  *(uint32_t *)PTR_BYTE_OFFSET(data.get(), offset) = len;
  offset += sizeof(uint32_t);
  std::memcpy(PTR_BYTE_OFFSET(data.get(), offset), edges, len * sizeof(edge_elem_t));
  memcpy_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mstart).count();
  comm->am_post(dstrank, am_id, std::move(data), data_len);
}

void NeighborSampler::scatter(Communicator *comm, uint16_t depth, uint64_t req_id, std::vector<node_id_t> &&seeds, uint64_t ppt) {
  CHECK(ppt > 0);
  uint64_t rem_ppt = ppt;
  auto estart = std::chrono::system_clock::now();
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
  erase_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - estart).count();
  std::minstd_rand0 engine(req_id ^ ppt ^ depth);
  std::vector<uint32_t> seq;
  if (fanouts[depth] > 0) {
    // https://github.com/python/cpython/blob/main/Lib/random.py
    size_t setsize = 21;
    if (fanouts[depth] > 5) {
      size_t ss = 1;
      for (size_t b = 4; b <= fanouts[depth] * 3; b *= 4) {
        ss *= 4;
      }
      setsize += ss;
    }
    seq.reserve(setsize);
  }
  uint32_t l, r = 0;
  for (uint16_t dstrank = 0; dstrank < size; dstrank++) {
    l = r;
    while (r < seeds.size() && seeds[r] / node_slit == dstrank) {
      r++;
    }
    if (l == r) continue;
    // LOG(INFO) << "scatter depth=" << depth <<  " rank=" << rank << " dstrank=" << dstrank << "req_id" << req_id;
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
      node_id_t *src_ids;
      size_t edge_len;
      node_id_t src;
      std::vector<node_id_t> next_seeds;
      size_t prev_edges_len = prog_que[req_id].edges[depth].size();
      for (uint32_t dst_idx = l; dst_idx < r; dst_idx++) {
        edge_shard.in_edges(&src_ids, &edge_len, seeds[dst_idx]);
        // LOG(INFO) << "dst_idx= " << dst_idx << " edge_len=" << edge_len << "depth=" << depth << "fanouts[depth]= " << fanouts[depth];
        if (fanouts[depth] < 0 || edge_len <= fanouts[depth]) {
          for (uint32_t idx = 0; idx < edge_len; idx++) {
            src = src_ids[idx];
            // LOG(INFO) << "self all: idx=" << idx << " src=" << src << " dst=" << dst;
            prog_que[req_id].edges[depth].push_back(edge_elem_t{src, seeds[dst_idx]});
            next_seeds.push_back(src);
          }
        } else {
          // sampling
          // auto mstart = std::chrono::system_clock::now();
          if (edge_len <= seq.capacity()) {
            seq.resize(edge_len);
            std::iota(seq.begin(), seq.begin() + edge_len, 0);
            for (uint32_t idx = edge_len-1; idx >= fanouts[depth]; idx--) {
              std::swap(seq[idx], seq[engine() % idx]);
            }
          } else {
            seq.clear();
            while (seq.size() < fanouts[depth]) {
              uint32_t idx = engine() % edge_len;
              if (std::find(seq.begin(), seq.end(), idx) == seq.end()) {
                seq.push_back(idx);
              }
            }
          }
          std::sort(seq.begin(), seq.begin() + fanouts[depth]);
          // memcpy_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mstart).count();
          for (uint32_t idx = 0; idx < fanouts[depth]; idx++) {
            src = src_ids[seq[idx]];
            // LOG(INFO) << "self sample: idx=" << idx << " src=" << src << " dst=" << dst;
            prog_que[req_id].edges[depth].push_back(edge_elem_t{src, seeds[dst_idx]});
            next_seeds.push_back(src);
          }
        }
      }
      std::inplace_merge(
        prog_que[req_id].edges[depth].begin(),
        prog_que[req_id].edges[depth].begin() + prev_edges_len,
        prog_que[req_id].edges[depth].end());
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
      std::vector<node_id_t> next_seeds;
      node_id_t *src_ids;
      size_t edge_len;
      for (uint32_t dst_idx = l; dst_idx < r; dst_idx++) {
        edge_shard.in_edges(&src_ids, &edge_len, seeds[dst_idx]);
        // LOG(INFO) << "dst_idx= " << dst_idx << " edge_len=" << edge_len << "depth=" << depth << "fanouts[depth]= " << fanouts[depth];
        if (fanouts[depth] < 0 || edge_len <= fanouts[depth]) {
          for (uint32_t idx = 0; idx < edge_len; idx++) {
            edge_elem_t elem{.src = src_ids[idx],.dst = seeds[dst_idx]};
            // LOG(INFO) << "other all: idx=" << idx << " src=" << elem.src << " dst=" << elem.dst;
            next_seeds.push_back(elem.src);
            edges.push_back(std::move(elem));
          }
        } else {
          // sampling
          // auto mstart = std::chrono::system_clock::now();
          if (edge_len <= seq.capacity()) {
            seq.resize(edge_len);
            std::iota(seq.begin(), seq.begin() + edge_len, 0);
            for (uint32_t idx = edge_len-1; idx >= fanouts[depth]; idx--) {
              std::swap(seq[idx], seq[engine() % idx]);
            }
          } else {
            seq.clear();
            while (seq.size() < fanouts[depth]) {
              uint32_t idx = engine() % edge_len;
              if (std::find(seq.begin(), seq.end(), idx) == seq.end()) {
                seq.push_back(idx);
              }
            }
          }
          std::sort(seq.begin(), seq.begin() + fanouts[depth]);
          // memcpy_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mstart).count();
          for (uint32_t idx = 0; idx < fanouts[depth]; idx++) {
            edge_elem_t elem{.src = src_ids[seq[idx]], .dst = seeds[dst_idx]};
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
  CHECK(r == seeds.size());
}

void inline NeighborSampler::recv_query(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint32_t len, const void *buffer) {
  auto mstart = std::chrono::system_clock::now();
  std::vector<node_id_t> seeds(len);
  std::memcpy(seeds.data(), buffer, sizeof(node_id_t) * len);
  memcpy_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mstart).count();
  scatter(comm, depth, req_id, std::move(seeds), ppt);
}

void inline NeighborSampler::enqueue(uint64_t req_id) {
  auto bstart = std::chrono::system_clock::now();
  std::vector<node_id_t> nodes(prog_que[req_id].seeds.NumElements());
  std::memcpy(&nodes[0], prog_que[req_id].seeds->data, sizeof(node_id_t) * prog_que[req_id].seeds.NumElements());
  HeteroGraphPtr a;
  std::vector<dgl::IdArray> b;
  blocks_with_label_t ret = {
    .blocks = std::vector<HeteroGraphPtr>{},
    .seeds = std::move(prog_que[req_id].seeds),
    .labels = std::move(prog_que[req_id].labels),
  };
  auto t1 = std::chrono::system_clock::now();
  std::vector<IdArray> dst_nodes{IdArray::FromVector(nodes)};
  create_idarray_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t1).count();
  for (uint16_t depth = 0; depth < num_layers; depth++) {
    edges_t edges = std::move(prog_que[req_id].edges[depth]);
    auto t0 = std::chrono::system_clock::now();
    auto edges_end = std::unique(edges.begin(), edges.end());
    erase_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t0).count();
    int64_t edges_len = edges_end - edges.begin();
    CHECK(edges_len >= 0);
    auto t2 = std::chrono::system_clock::now();
    IdArray src = IdArray::Empty(std::vector<int64_t>{edges_len},  DLDataType{kDLInt, 8 * sizeof(node_id_t), 1}, DLContext{kDLCPU, 0});
    IdArray dst = IdArray::Empty(std::vector<int64_t>{edges_len},  DLDataType{kDLInt, 8 * sizeof(node_id_t), 1}, DLContext{kDLCPU, 0});
    create_idarray_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t2).count();
    auto t3 = std::chrono::system_clock::now();
    for (int64_t idx = 0; idx < edges_len; idx++) {
      *(node_id_t *)PTR_BYTE_OFFSET(src->data, sizeof(node_id_t) * idx) = edges[idx].src;
      *(node_id_t *)PTR_BYTE_OFFSET(dst->data, sizeof(node_id_t) * idx) = edges[idx].dst;
    }
    edge_copy_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t3).count();
    auto t4 = std::chrono::system_clock::now();
    HeteroGraphPtr g = CreateFromCOO(1, edges_len, edges_len, src, dst, false, false);
    create_graph_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t4).count();
    auto t5 = std::chrono::system_clock::now();
    std::vector<IdArray> src_nodes;
    create_idarray_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t5).count();
    auto t6 = std::chrono::system_clock::now();
    std::tie(a, b) = transform::ToBlock<kDLCPU, node_id_t>(g, dst_nodes, true, &src_nodes);
    toblock_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t6).count();
    ret.blocks.push_back(std::move(a));
    dst_nodes = std::move(src_nodes);
  }
  std::reverse(ret.blocks.begin(), ret.blocks.end());
  auto t7 = std::chrono::system_clock::now();
  CHECK(dst_nodes.size() == 1);
  ret.input_nodes = std::move(dst_nodes[0]);
  erase_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t7).count();
  output_que->push(std::move(ret));
  prog_que.erase(req_id);
  /*
  LOG(INFO) << "sampler: req_id=" << req_id << " is finished"
    << ",total_edges=" << total_edges
    << ",prog_que.size()=" << prog_que.size();
  */
  auto bend = std::chrono::system_clock::now();
  build_block_time += std::chrono::duration_cast<std::chrono::milliseconds>(bend - bstart).count();
}

void inline NeighborSampler::recv_response(Communicator *comm, uint16_t depth, uint64_t ppt, uint64_t req_id, uint32_t len, const void *buffer) {
  edges_t &edges = prog_que[req_id].edges[depth];
  size_t edges_len = edges.size();
  edges.insert(edges.end(), (edge_elem_t *)buffer, (edge_elem_t *)buffer + len);
  std::inplace_merge(edges.begin(), edges.begin() + edges_len, edges.end());
  prog_que[req_id].ppt += ppt;
  // LOG(INFO) << "req_id=" << req_id << " ppt=" << ppt;
  if (prog_que[req_id].ppt == PPT_ALL) {
    enqueue(req_id);
  }
}

void NeighborSampler::am_recv(Communicator *comm, const void *buffer, size_t length) {
  size_t offset = 0;
  uint64_t shifted_id;
  uint16_t depth;
  uint32_t data_length;
  uint64_t ppt;
  auto rstart = std::chrono::system_clock::now();
  shifted_id = *(uint64_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint64_t);
  ppt = *(uint64_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint64_t);
  depth = *(uint16_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint16_t);
  data_length = *(uint32_t *)PTR_BYTE_OFFSET(buffer, offset);
  offset += sizeof(uint32_t);

  if (shifted_id & 1) {
    recv_response(comm, depth, ppt, shifted_id>>1, data_length, PTR_BYTE_OFFSET(buffer, offset));
    CHECK(offset + sizeof(edge_elem_t) * data_length == length);
  } else {
    auto qstart = std::chrono::system_clock::now();
    recv_query(comm, depth, ppt, shifted_id>>1, data_length, PTR_BYTE_OFFSET(buffer, offset));
    CHECK(offset + sizeof(node_id_t) * data_length == length);
    recv_query_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - qstart).count();
  }
  auto rend = std::chrono::system_clock::now();
  handle_req_time += std::chrono::duration_cast<std::chrono::milliseconds>(rend - rstart).count();
}

unsigned NeighborSampler::progress(Communicator *comm) {
  seed_with_label_t input;
  if (input_que->try_dequeue(input)) {
    CHECK(input.seeds->dtype.code == kDLInt);
    CHECK(input.seeds->dtype.bits == 8 * sizeof(node_id_t));
    auto sstart = std::chrono::system_clock::now();
    std::vector<node_id_t> seeds(input.seeds.NumElements());
    std::memcpy(&seeds[0], input.seeds->data, sizeof(node_id_t) * input.seeds.NumElements());
    prog_que[req_id] = neighbor_sampler_prog_t(num_layers, std::move(input.seeds), std::move(input.labels));
#ifdef DGL_USE_NVTX
  nvtxRangePush(__FUNCTION__);
  nvtxMark("NeighborSampler::progress(): Scatter");
#endif // DGL_USE_NVTX
    scatter(comm, 0, req_id, std::move(seeds), PPT_ALL);
#ifdef DGL_USE_NVTX
    nvtxRangePop();
#endif // DGL_USE_NVTX
    req_id += size;
    auto send = std::chrono::system_clock::now();
    self_scatter_time += std::chrono::duration_cast<std::chrono::milliseconds>(send - sstart).count();
    return 1;
  }
  return 0;
}

FeatLoader::FeatLoader(feat_loader_arg_t &&arg,
  std::queue<blocks_with_label_t>  *input_que,
  BlockingConcurrentQueue<blocks_with_feat_t> *output_que)
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
  size_t length = local_feats->dtype.bits / 8;
  for (int dim = 0; dim < local_feats->ndim; dim++) {
    length *= local_feats->shape[dim];
  }
  return std::make_pair(local_feats->data, length);
}

void FeatLoader::enqueue(uint64_t req_id) {
  std::vector<dgl::HeteroGraphPtr> blocks = std::move(prog_que[req_id].inputs.blocks);
  NDArray labels = std::move(prog_que[req_id].inputs.labels);
  NDArray seeds = std::move(prog_que[req_id].inputs.seeds);
  NDArray feats = std::move(prog_que[req_id].feats);
  batch_num_total++;
  batch_size_total += feats.NumElements() * (feats->dtype.bits / 8); 
  prog_que.erase(req_id);
  blocks_with_feat_t ret = {
    .labels = std::move(labels),
    .feats = std::move(feats),
    .seeds = std::move(seeds),
    .blocks = std::move(blocks),
  };
  output_que->enqueue(std::move(ret));
}

unsigned FeatLoader::progress(Communicator *comm) {
  if (!input_que->empty()) {
    auto sstart = std::chrono::system_clock::now();
    blocks_with_label_t item = std::move(input_que->front());
    prog_que[req_id] = feat_loader_prog_t(std::move(item));
    input_que->pop();
    NDArray input_nodes = std::move(prog_que[req_id].inputs.input_nodes);
    std::vector<int64_t> shape{
      static_cast<int64_t>(input_nodes.NumElements()),
      static_cast<int64_t>(feats_row)
    };
    prog_que[req_id].feats = pool.alloc(shape, local_feats->dtype);
    for (size_t row = 0; row < shape[0]; row++) {
      node_id_t node = input_nodes.Ptr<node_id_t>()[row];
      int src_rank = node / node_slit;
      CHECK(src_rank < size);
      uint64_t offset = (node % node_slit) * feats_row_size;
      void *recv_buffer = PTR_BYTE_OFFSET(prog_que[req_id].feats->data, feats_row_size * row);
      if (src_rank == rank) {
        std::memcpy(recv_buffer, PTR_BYTE_OFFSET(served_buffer().first, offset), feats_row_size);
        prog_que[req_id].received++;
        if (prog_que[req_id].received == prog_que[req_id].num_input_nodes) {
          // LOG(INFO) << "req_id=" << req_id << " is completed";
          enqueue(req_id);
        }
        continue;
      }
      comm->rma_read(src_rank, rma_id, req_id, recv_buffer, offset, feats_row_size);
    }
    req_id += size;
    memcpy_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - sstart).count();
    return 1;
  }
  return 0;
}

void FeatLoader::rma_read_cb(Communicator *comm, uint64_t req_id) {
  prog_que[req_id].received++;
  if (prog_que[req_id].received == prog_que[req_id].num_input_nodes) {
    /*
    LOG(INFO) << "rma_read_cb: req_id=" << req_id << " is finished"
      << ",num_input_nodes=" << prog_que[req_id].num_input_nodes
      << ",prog_que.size()=" << prog_que.size();
    */
    enqueue(req_id);
  }
}


NodeDataLoader::NodeDataLoader(Communicator *comm, node_dataloader_arg_t &&arg)
: ServiceManager(arg.rank, arg.size, comm), rank(arg.rank), size(arg.size) {
  neighbor_sampler_arg_t arg0 = {
    .rank = arg.rank,
    .size = arg.size,
    .num_nodes = arg.num_nodes,
    .num_layers = arg.num_layers,
    .fanouts = std::move(arg.fanouts),
    .edge_shard = std::move(arg.edge_shard),
  };
  std::unique_ptr<NeighborSampler> sampler(new NeighborSampler(std::move(arg0), &input_que, &bridge_que));
  add_am_service(std::move(sampler));
  feat_loader_arg_t arg1 = {
    .rank = arg.rank,
    .size = arg.size,
    .num_nodes = arg.num_nodes,
    .local_feats = std::move(arg.local_feats),
  };
  std::unique_ptr<FeatLoader> loader(new FeatLoader(std::move(arg1), &bridge_que, &output_que));
  add_rma_service(std::move(loader));
}

void NodeDataLoader::enqueue(seed_with_label_t &&item) {
  input_que.enqueue(std::move(item));
}

void NodeDataLoader::dequeue(blocks_with_feat_t &item) {
  if (output_que.size_approx() == 0) {
    LOG(INFO) << "rank=" << rank << " might be starved in dequeue";
  }
  output_que.wait_dequeue(item);
}

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  CommunicatorRef comm = args[0];
  int num_layers = args[1];
  int num_nodes = args[2];
  List<Value> _fanouts = args[3];
  std::vector<int> fanouts(ListValueToVector<int>(_fanouts));
  NDArray local_feats = args[4];
  NDArray src = args[5];
  NDArray dst = args[6];
  node_dataloader_arg_t arg = {
    .rank = comm->rank,
    .size = comm->size,
    .num_nodes = num_nodes,
    .num_layers = num_layers,
    .fanouts = fanouts,
    .local_feats = std::move(local_feats),
    .edge_shard = edge_shard_t(std::move(src),std::move(dst), comm->rank, comm->size, num_nodes),
  };
  std::shared_ptr<NodeDataLoader> loader(new NodeDataLoader(comm.sptr().get(), std::move(arg)));
  *rv = loader;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2EnqueueToNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  NDArray seeds = args[1];
  NDArray labels = args[2];
  seed_with_label_t item = {
    .seeds = std::move(seeds),
    .labels = std::move(labels),
  };
  loader->enqueue(std::move(item));
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2DequeueToNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  blocks_with_feat_t item;
  auto dstart = std::chrono::system_clock::now();
#ifdef DGL_USE_NVTX
  nvtxRangePush(__FUNCTION__);
  nvtxMark("Dequeue");
#endif // DGL_USE_NVTX
  loader->dequeue(item);
#ifdef DGL_USE_NVTX
  nvtxRangePop();
#endif // DGL_USE_NVTX
  auto dend = std::chrono::system_clock::now();
  if (dequeue_time < 0) {
    LOG(INFO) << "first dequeue_time= " << std::chrono::duration_cast<std::chrono::milliseconds>(dend - dstart).count(); 
    dequeue_time = 0;
  }
  dequeue_time += std::chrono::duration_cast<std::chrono::milliseconds>(dend - dstart).count();
  List<Value> ret;
  List<Value> blocks;
  // LOG(INFO) << "item.blocks.size()=" << item.blocks.size();
  CHECK(item.blocks.size() > 0);
  for (size_t depth = 0; depth < item.blocks.size(); depth++) {
    blocks.push_back(Value(MakeValue(HeteroGraphRef(item.blocks[depth]))));
  }
  ret.push_back(Value(MakeValue(std::move(blocks))));
  ret.push_back(Value(MakeValue(std::move(item.labels))));
  ret.push_back(Value(MakeValue(std::move(item.feats))));
  *rv = ret;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2MapRMAService")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  rma_mem_ret_t meta = loader->map_rma_service();
  List<Value> ret;
  ret.push_back(Value(MakeValue(meta.rkeybuf)));
  ret.push_back(Value(MakeValue(meta.rkeybuf_len)));
  ret.push_back(Value(MakeValue(meta.address)));
  ret.push_back(Value(MakeValue(meta.address_len)));
  *rv = ret;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2PrepareRMAService")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  std::string rkeybufs = args[1];
  std::string addrs = args[2];
  loader->prepare_rma_service(&rkeybufs[0], rkeybufs.size(), &addrs[0], addrs.size());
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2LaunchNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  loader->launch();
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2TermNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  loader->terminate();
  LOG(INFO)
    << " rank=" << loader->rank
    << " dequeue_time=" << dequeue_time
    << " build_block_time=" << build_block_time
    << " self_scatter_time=" << self_scatter_time
    << " handle_req_time=" << handle_req_time
    << " recv_query_time=" << recv_query_time
    << " create_idarray_time=" << create_idarray_time
    << " toblock_time=" << toblock_time
    << " create_graph_time" << create_graph_time
    << " edge_copy_time" << edge_copy_time
    << " erase_time=" << erase_time
    << " memcpy_time=" << memcpy_time
    << " batch_num_total=" << batch_num_total
    << " batch_size_total=" << batch_size_total;
});

}
}