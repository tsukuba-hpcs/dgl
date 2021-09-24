/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/context.cc
 * \brief Implementation of HPC context.
 */

#include "context.h"

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include <mpi.h>
#include <memory>
#include <vector>
#include <string>

#include "../c_api_common.h"



namespace dgl {
namespace hpc {
namespace context {

using namespace dgl::runtime;

//////////////////////////// C APIs ////////////////////////////

//////////////////////////// Context ////////////////////////////

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCCreateContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int flag;
  ucs_status_t status;
  MPI_Initialized(&flag);
  if (flag) {
    LOG(FATAL) << "MPI is already Initialized";
  }
  MPI_Init(NULL, NULL);
  std::shared_ptr<Context> rst(new Context);
  MPI_Comm_rank(MPI_COMM_WORLD, &rst->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rst->size);
  ucp_params_t ucp_params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES,
    .features = UCP_FEATURE_RMA,
  };
  status = ucp_init(&ucp_params, NULL, &rst->ucp_context);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_init failed with " << ucs_status_string(status);
  }
  ucp_worker_params_t worker_params = {
    .field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
    .thread_mode = UCS_THREAD_MODE_SINGLE,
  };
  status = ucp_worker_create(rst->ucp_context,
    &worker_params, &rst->ucp_worker);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_worker_create failed with " << ucs_status_string(status);
  }
  *rv = rst;
});

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCFinalizeContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  for (ucp_ep_h &ep : ctx->remote_ep) {
    ucp_ep_destroy(ep);
  }
  ucp_worker_destroy(ctx->ucp_worker);
  ucp_cleanup(ctx->ucp_context);
  MPI_Finalize();
});

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCContextGetRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const ContextRef ctx = args[0];
  *rv = ctx->rank;
});

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCContextGetSize")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const ContextRef ctx = args[0];
  *rv = ctx->size;
});

//////////////////////////// Manager ////////////////////////////

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

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCManagerLaunchWorker")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  int32_t num_workers = args[1];
  std::vector<std::string> worker_args;
  for (int i = 2; i < args.num_args; i++) {
    const std::string arg = args[i];
    worker_args.push_back(arg);
  }
  spawn_worker(&ctx->inter_comm, num_workers, worker_args);
  ctx->remote_rank = -1;
  ctx->remote_size = num_workers;
});

static inline void bcast_manager_context(ContextRef ctx) {
  MPI_Barrier(ctx->inter_comm);
  MPI_Bcast(&ctx->rank, 1, MPI_INT32_T, MPI_ROOT, ctx->inter_comm);
  MPI_Bcast(&ctx->size, 1, MPI_INT32_T, MPI_ROOT, ctx->inter_comm);
}

static inline void bcast_manager_address(ContextRef ctx) {
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

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCManagerServe")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  bcast_manager_context(ctx);
  bcast_manager_address(ctx);
});

//////////////////////////// Worker ////////////////////////////

static inline void recv_manager_context(ContextRef ctx) {
  MPI_Barrier(ctx->inter_comm);
  MPI_Bcast(&ctx->remote_rank, 1, MPI_INT32_T, 0, ctx->inter_comm);
  MPI_Bcast(&ctx->remote_size, 1, MPI_INT32_T, 0, ctx->inter_comm);
  LOG(INFO) << "remote_rank=" << ctx->remote_rank << " "
            << "remote_size=" << ctx->remote_size;
}

static inline void recv_manager_address(ContextRef ctx) {
  size_t addr_len;
  ucs_status_t status;
  ctx->remote_ep.resize(ctx->remote_size);
  MPI_Bcast(&addr_len, sizeof(size_t), MPI_BYTE, 0, ctx->inter_comm);
  std::vector<char> buffer(addr_len * ctx->remote_size);
  MPI_Bcast(&buffer[0], addr_len * ctx->remote_size, MPI_BYTE, 0, ctx->inter_comm);
  for (int32_t rank = 0; rank < ctx->remote_size; rank++) {
    ucp_ep_params_t ep_params = {
      .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS,
      .address = reinterpret_cast<ucp_address_t *>(&buffer[addr_len * rank]),
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

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCWorkerConnect")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  MPI_Comm_get_parent(&ctx->inter_comm);
  recv_manager_context(ctx);
  recv_manager_address(ctx);
});

}  // namespace context
}  // namespace hpc
}  // namespace dgl
