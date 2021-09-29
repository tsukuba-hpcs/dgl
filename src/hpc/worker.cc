/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/worker.cc
 * \brief Implementation of HPC worker.
 */

#include <mpi.h>

#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/packed_func_ext.h>

#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "context.h"
#include "worker.h"

#include "../c_api_common.h"

namespace dgl {
namespace hpc {
namespace worker {

using namespace dgl::runtime;

//////////////////////////// Context ////////////////////////////

static inline void recv_manager_context(context::ContextRef ctx) {
  context::barrier(ctx, ctx->inter_comm);
  MPI_Bcast(&ctx->remote_rank, 1, MPI_INT32_T, 0, ctx->inter_comm);
  MPI_Bcast(&ctx->remote_size, 1, MPI_INT32_T, 0, ctx->inter_comm);
  LOG(INFO) << "remote_rank=" << ctx->remote_rank << " "
            << "remote_size=" << ctx->remote_size;
}

static void ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
  context::Context* ctx = reinterpret_cast<context::Context*>(arg);
  LOG(INFO) << "endpoint error " << ucs_status_string(status);
}

static inline void recv_manager_address(context::ContextRef ctx) {
  size_t addr_len;
  ucs_status_t status;
  ctx->remote_ep.resize(ctx->remote_size);
  MPI_Bcast(&addr_len, sizeof(size_t), MPI_BYTE, 0, ctx->inter_comm);
  std::vector<char> buffer(addr_len * ctx->remote_size);
  MPI_Bcast(&buffer[0], addr_len * ctx->remote_size, MPI_BYTE, 0, ctx->inter_comm);
  for (int32_t rank = 0; rank < ctx->remote_size; rank++) {
    ucp_ep_params_t ep_params = {
      .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                    UCP_EP_PARAM_FIELD_ERR_HANDLER,
      .address = reinterpret_cast<ucp_address_t *>(&buffer[addr_len * rank]),
      .err_mode = UCP_ERR_HANDLING_MODE_PEER,
      .err_handler = {
        .cb = ep_err_cb,
        .arg = reinterpret_cast<void*>(ctx.sptr().get()),
      }
    };
    status = ucp_ep_create(ctx->ucp_worker, &ep_params, &ctx->remote_ep[rank]);
    if (status != UCS_OK) {
      LOG(FATAL) << "parent_manager_rank=" << ctx->remote_rank << " "
                 << "worker_rank=" << ctx->rank << " "
                 << "target_manager_rank=" << rank << " "
                 << "ucp_ep_create failed " << ucs_status_string(status);
    }
  }
}

static inline void recv_shard_metadata(context::ContextRef ctx, ShardClientRef client) {
  ucs_status_t status;
  int size;
  int name_len;
  std::vector<char> name_buffer;
  MPI_Bcast(&size, 1, MPI_INT, 0, ctx->inter_comm);
  client->metadata.resize(size);
  int64_t rkeys_len_total;
  std::vector<char> rkeys_buffer_total;
  std::vector<int> displs(ctx->remote_size);
  std::vector<void*> data_total(ctx->remote_size);
  for (int id = 0; id < size; id++) {
    MPI_Bcast(&name_len, 1, MPI_INT, 0, ctx->inter_comm);
    name_buffer.resize(name_len);
    MPI_Bcast(&name_buffer[0], name_len, MPI_CHAR, 0, ctx->inter_comm);
    std::string name(name_buffer.begin(), name_buffer.end());
    client->name2id[name] = id;
    MPI_Bcast(&client->metadata[id].dtype, sizeof(DGLType), MPI_BYTE, 0, ctx->inter_comm);
    MPI_Bcast(&client->metadata[id].col_ndim, 1, MPI_INT, 0, ctx->inter_comm);
    if (client->metadata[id].col_ndim > 0) {
      client->metadata[id].col_shape.resize(client->metadata[id].col_ndim);
      MPI_Bcast(&client->metadata[id].col_shape[0],
        client->metadata[id].col_ndim, MPI_INT64_T, 0, ctx->inter_comm);
      client->metadata[id].row_length = client->metadata[id].dtype.bits / 8;
      for (int len : client->metadata[id].col_shape) {
        client->metadata[id].row_length *= len;
      }
    } else {
      client->metadata[id].col_shape.resize(1);
      client->metadata[id].col_shape[0] = 1;
      client->metadata[id].row_length = client->metadata[id].dtype.bits / 8;
    }
    LOG(INFO) << "id=" << id << " "
              << "name=" << name << " "
              << "col_ndim=" << client->metadata[id].col_ndim;
    MPI_Bcast(&rkeys_len_total, 1, MPI_INT64_T, 0, ctx->inter_comm);
    rkeys_buffer_total.resize(rkeys_len_total);
    MPI_Bcast(&displs[0], ctx->remote_size, MPI_INT, 0, ctx->inter_comm);
    MPI_Bcast(&rkeys_buffer_total[0], rkeys_len_total, MPI_BYTE, 0, ctx->inter_comm);
    MPI_Bcast(&data_total[0], sizeof(void*) * ctx->remote_size, MPI_BYTE, 0, ctx->inter_comm);
    client->metadata[id].data.clear();
    client->metadata[id].rkeys.clear();
    for (int rank = 0; rank < ctx->remote_size; rank++) {
      ucp_rkey_h rkey;
      client->metadata[id].data.push_back(std::move(data_total[rank]));
      status = ucp_ep_rkey_unpack(ctx->remote_ep[rank],
        &rkeys_buffer_total[displs[rank]], &rkey);
      if (status != UCS_OK) {
        LOG(FATAL) << "ucp_ep_rkey_unpack failed with " << ucs_status_string(status);
      }
      client->metadata[id].rkeys.push_back(std::move(rkey));
    }
  }
}

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCWorkerConnect")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  context::ContextRef ctx = args[0];
  MPI_Comm_get_parent(&ctx->inter_comm);
  recv_manager_context(ctx);
  recv_manager_address(ctx);
});

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCCreateShardClient")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  std::shared_ptr<ShardClient> rst(new ShardClient);
  LOG(INFO) << "CreateShardClient called";
  *rv = rst;
});

//////////////////////////// Shard ////////////////////////////

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCWorkerRecvMetadata")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  context::ContextRef ctx = args[0];
  ShardClientRef client = args[1];
  recv_shard_metadata(ctx, client);
});

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCFinalizeShardClient")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ShardClientRef client = args[0];
  for (TensorMetaData &metadata : client->metadata) {
    for (ucp_rkey_h rkey : metadata.rkeys) {
      ucp_rkey_destroy(rkey);
    }
  }
});

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCGetTensorIDFromName")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ShardClientRef client = args[0];
  std::string name = args[1];
  if (client->name2id.count(name) == 0) {
    LOG(FATAL) << "name=" << name << " is not found";
  }
  int id = client->name2id[name];
  *rv = id;
});

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCGetTensorShapeFromID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ShardClientRef client = args[0];
  int id = args[1];
  List<Value> ret;
  for (int d = 0; d < client->metadata[id].col_ndim; d++) {
    ret.push_back(Value(MakeValue(client->metadata[id].col_shape[d])));
  }
  *rv = ret;
});

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCGetTensorDtypeFromID")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ShardClientRef client = args[0];
  int id = args[1];
  *rv = client->metadata[id].dtype;
});


//////////////////////////// SlicePool ////////////////////////

SlicePool::SlicePool(int pool_size, const TensorMetaData *metadata) :
  pool_size(pool_size), head(0), metadata(metadata) {
  if (pool_size <= 0) {
    LOG(FATAL) << "pool_size=" << pool_size << " is invalid";
  }
  used.assign(pool_size, false);
  for (int i = 0; i < pool_size; i++) {
    NDArray s = NDArray::Empty(metadata->col_shape, metadata->dtype,
      DLContext{DLDeviceType::kDLCPU, 0});
    slice.push_back(std::move(s));
  }
}

NDArray* SlicePool::alloc() {
  int cur = head;
  do {
    if (!used[cur]) {
      used[cur] = true;
      head = (cur + 1) % pool_size;
      return &slice[cur];
    }
    cur = (cur + 1) % pool_size;
  } while (cur != head);
  LOG(FATAL) << "SlicePool alloc() failed";
  return NULL;
}

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCAllocSlicePool")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ShardClientRef client = args[0];
  int pool_size = args[1];
  for (int id = 0; id < client->name2id.size(); id++) {
    client->pool.emplace_back(pool_size, &client->metadata[id]);
  }
});

DGL_REGISTER_GLOBAL("hpc.worker._CAPI_HPCFetchSlice")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  context::ContextRef ctx = args[0];
  ShardClientRef client = args[1];
  int id = args[2];
  int rank = args[3];
  int row = args[4];
  NDArray *buffer = client->pool[id].alloc();
  ucs_status_t stat;
  size_t row_length = client->metadata[id].row_length;
  stat = ucp_get(ctx->remote_ep[rank],
    buffer->ToDLPack()->dl_tensor.data,
    row_length,
    (uint64_t)client->metadata[id].data[rank] + row * row_length,
    client->metadata[id].rkeys[rank]);
  if (stat != UCS_OK) {
    LOG(FATAL) << "ucp_get failed";
  }
  *rv = *buffer;
});

}  // namespace worker
}  // namespace hpc
}  // namespace dgl
