/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/manager.cc
 * \brief Implementation of HPC manager.
 */

#include <mpi.h>

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "context.h"
#include "manager.h"

#include "../c_api_common.h"


namespace dgl {
namespace hpc {
namespace manager {

using namespace dgl::runtime;

//////////////////////////// Context ////////////////////////////

static inline void spawn_worker(MPI_Comm *comm, int32_t num_workers,
                                const std::vector<std::string> &worker_args) {
  if (num_workers < 1) {
    LOG(FATAL) << "num_workers = " << num_workers << " is invalid";
  }
  MPI_Info info;
  char hostname[30];
  int hostname_len;
  std::vector<std::vector<char>> args_buffer;
  for (size_t i=1; i < worker_args.size(); i++) {
    args_buffer.emplace_back(worker_args[i].begin(), worker_args[i].end());
    args_buffer.back().push_back('\0');
  }
  std::vector<char *> args;
  for (std::vector<char> &buf : args_buffer) {
    args.push_back(&buf[0]);
  }
  args.push_back(NULL);
  MPI_Get_processor_name(hostname, &hostname_len);
  LOG(INFO) << "host=" << hostname << " spawn " << num_workers << " workers";
  MPI_Info_create(&info);
  MPI_Info_set(info, "host", hostname);
  MPI_Comm_spawn(worker_args[0].c_str(), &args[0], num_workers, info, 0,
    MPI_COMM_SELF, comm, MPI_ERRCODES_IGNORE);
}

DGL_REGISTER_GLOBAL("hpc.manager._CAPI_HPCManagerLaunchWorker")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  context::ContextRef ctx = args[0];
  int32_t num_workers = args[1];
  std::vector<std::string> worker_args;
  for (int i = 2; i < args.num_args; i++) {
    const std::string arg = args[i];
    worker_args.push_back(arg);
  }
  spawn_worker(&ctx->inter_comm, num_workers, worker_args);
  ctx->remote_size = num_workers;
});

static inline void bcast_manager_context(context::ContextRef ctx) {
  MPI_Barrier(ctx->inter_comm);
  MPI_Bcast(&ctx->rank, 1, MPI_INT32_T, MPI_ROOT, ctx->inter_comm);
  MPI_Bcast(&ctx->size, 1, MPI_INT32_T, MPI_ROOT, ctx->inter_comm);
}

static inline void bcast_manager_address(context::ContextRef ctx) {
  ucp_address_t *addr;
  size_t addr_len;
  ucs_status_t status;
  status = ucp_worker_get_address(ctx->ucp_worker, &addr, &addr_len);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_worker_get_address failed with " << ucs_status_string(status);
  }
  LOG(INFO) << "ucp address length=" << addr_len;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&addr_len, sizeof(size_t), MPI_BYTE, MPI_ROOT, ctx->inter_comm);
  std::vector<char> buffer(addr_len * ctx->size);
  MPI_Allgather(addr, addr_len, MPI_BYTE, &buffer[0], addr_len, MPI_BYTE, MPI_COMM_WORLD);
  MPI_Bcast(&buffer[0], addr_len * ctx->size, MPI_BYTE, MPI_ROOT, ctx->inter_comm);
  ucp_worker_release_address(ctx->ucp_worker, addr);
}

static inline void register_shard(context::ContextRef ctx, ShardRef shard,
  void** rkeys, size_t *rkeys_len) {
  ucs_status_t status;
  ctx->register_mem.resize(shard->tensor.size());
  for (int id = 0; id < static_cast<int>(shard->tensor.size()); id++) {
    size_t byte_length = shard->tensor[id]->dtype.bits % 8;
    for (int j = 0; j < shard->tensor[id]->ndim; j++) {
      byte_length *= shard->tensor[id]->shape[j];
    }
    ucp_mem_map_params mem_params = {
      .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH,
      .address = shard->tensor[id]->data,
      .length = byte_length,
    };
    status = ucp_mem_map(ctx->ucp_context, &mem_params, &ctx->register_mem[id]);
    if (status != UCS_OK) {
      LOG(FATAL) << "rank=" << ctx->rank << " "
                 << "tensor_id=" << id << " "
                 << "ucp_mem_map failed with" << ucs_status_string(status);
    }
    status = ucp_rkey_pack(ctx->ucp_context, ctx->register_mem[id], &rkeys[id], &rkeys_len[id]);
    if (status != UCS_OK) {
      LOG(FATAL) << "rank=" << ctx->rank << " "
                 << "tensor_id=" << id << " "
                 << "ucp_rkey_pack failed with" << ucs_status_string(status);
    }
  }
}

static inline void bcast_manager_shard(context::ContextRef ctx, ShardRef shard,
  void** rkeys, size_t* rkeys_len) {
  int size = shard->tensor.size();
  if (static_cast<int>(shard->name2id.size()) != size) {
    LOG(FATAL) << "number of tensor is not equal to number of name";
  }
  MPI_Bcast(&size, 1, MPI_INT, MPI_ROOT, ctx->inter_comm);
  std::vector<char> name;
  bool found;
  int name_len;
  DLDataType dtype;
  int ndim;
  std::vector<int64_t> shape;
  int64_t rkey_len, rkeys_len_total;
  std::vector<char> rkeys_buffer_total;
  std::vector<int> recvcounts(ctx->size), displs(ctx->size);
  void *data;
  std::vector<void*> data_total(ctx->size);
  // for each tensor, broadcast these data.
  // 1. name
  // 2. dtype
  // 3. ndim
  // 4. shape
  // 5. rkeys total length (rkey is used for RDMA Read/Write to buffer)
  // 6. displs (offset of buffer) for each manager's rkey
  // 7. rkeys (rank=k manager's rkey is rkeys[displs[k]])
  // 8. buffer addresses for each manager. (buffer address is used for RDMA Read/Write)
  for (int id = 0; id < size; id++) {
    found = false;
    for (auto kv : shard->name2id) {
      if (kv.second == id) {
        found = true;
        name = std::vector<char>(kv.first.begin(), kv.first.end());
        break;
      }
    }
    if (!found) {
      LOG(FATAL) << "id=" << id << "'s name is not found in shard";
    }
    name_len = name.size();
    MPI_Bcast(&name_len, 1, MPI_INT, MPI_ROOT, ctx->inter_comm);
    MPI_Bcast(&name[0], name_len, MPI_CHAR, MPI_ROOT, ctx->inter_comm);
    std::memcpy(&dtype, &shard->tensor[id]->dtype, sizeof(DLDataType));
    ndim = shard->tensor[id]->ndim;
    shape.assign(shard->tensor[id]->shape, shard->tensor[id]->shape + ndim);
    MPI_Bcast(&dtype, sizeof(DGLType), MPI_BYTE, MPI_ROOT, ctx->inter_comm);
    MPI_Bcast(&ndim, 1, MPI_INT, MPI_ROOT, ctx->inter_comm);
    MPI_Bcast(&shape[0], ndim, MPI_INT64_T, MPI_ROOT, ctx->inter_comm);
    rkey_len = static_cast<int64_t>(rkeys_len[id]);
    MPI_Allreduce(&rkey_len, &rkeys_len_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    rkeys_buffer_total.resize(rkeys_len_total);
    MPI_Allgatherv(rkeys[id], rkey_len, MPI_BYTE, &rkeys_buffer_total[0],
      &recvcounts[0], &displs[0], MPI_BYTE, MPI_COMM_WORLD);
    MPI_Bcast(&rkeys_len_total, 1, MPI_INT64_T, MPI_ROOT, ctx->inter_comm);
    MPI_Bcast(&displs[0], ctx->size, MPI_INT, MPI_ROOT, ctx->inter_comm);
    MPI_Bcast(&rkeys_buffer_total[0], rkeys_len_total, MPI_BYTE, MPI_ROOT, ctx->inter_comm);
    data = shard->tensor[id]->data;
    MPI_Allgather(&data, sizeof(void*), MPI_BYTE, &data_total[0],
      sizeof(void*), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Bcast(&data_total[0], sizeof(void*) * ctx->size, MPI_BYTE, MPI_ROOT, ctx->inter_comm);
  }
}

static inline void release_rkeys(context::ContextRef ctx, ShardRef shard, void** rkeys) {
  for (int i = 0; i < static_cast<int>(shard->tensor.size()); i++) {
    ucp_rkey_buffer_release(rkeys[i]);
  }
}

DGL_REGISTER_GLOBAL("hpc.manager._CAPI_HPCManagerServe")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  context::ContextRef ctx = args[0];
  ShardRef shard = args[1];
  std::vector<void*> rkeys(shard->tensor.size());
  std::vector<size_t> rkeys_len(shard->tensor.size());
  register_shard(ctx, shard, &rkeys[0], &rkeys_len[0]);
  bcast_manager_context(ctx);
  bcast_manager_address(ctx);
  bcast_manager_shard(ctx, shard, &rkeys[0], &rkeys_len[0]);
  release_rkeys(ctx, shard, &rkeys[0]);
});

//////////////////////////// Shard ////////////////////////////

DGL_REGISTER_GLOBAL("hpc.manager._CAPI_HPCCreateShard")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  std::shared_ptr<Shard> rst(new Shard);
  LOG(INFO) << "CreateShard called";
  *rv = rst;
});

DGL_REGISTER_GLOBAL("hpc.manager._CAPI_HPCRegisterTensor")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ShardRef shard = args[0];
  std::string name = args[1];
  NDArray tensor = args[2];
  if (shard->name2id.count(name)) {
    LOG(FATAL) << name << " is already exists in Shard";
  }
  int id = shard->name2id.size();
  if (static_cast<int>(shard->tensor.size()) != id) {
    LOG(FATAL) << "number of tensor is not equal to number of name";
  }
  shard->name2id[name] = id;
  shard->tensor.push_back(std::move(tensor));
  *rv = id;
});

}  // namespace manager
}  // namespace hpc
}  // namespace dgl
