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
#include <utility>

#include "../c_api_common.h"



namespace dgl {
namespace hpc {
namespace context {

using namespace dgl::runtime;

//////////////////////////// C APIs ////////////////////////////

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCCreateContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int flag;
  ucs_status_t status;
  MPI_Initialized(&flag);
  if (flag) {
    LOG(FATAL) << "MPI is already Initialized";
  }
  MPI_Init(NULL, NULL);
  std::shared_ptr<Context> ctx(new Context);
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);
  ctx->inter_comm = MPI_COMM_WORLD;
  ctx->remote_rank = -1;
  ctx->remote_size = 0;
  ucp_params_t ucp_params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES,
    .features = UCP_FEATURE_RMA,
  };
  status = ucp_init(&ucp_params, NULL, &ctx->ucp_context);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_init failed with " << ucs_status_string(status);
  }
  ucp_worker_params_t worker_params = {
    .field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
    .thread_mode = UCS_THREAD_MODE_SINGLE,
  };
  status = ucp_worker_create(ctx->ucp_context,
    &worker_params, &ctx->ucp_worker);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_worker_create failed with " << ucs_status_string(status);
  }
  *rv = ctx;
});

static inline void sync_all_proc(ContextRef ctx) {
  MPI_Request req1, req2;
  int flag;
  // If Manager
  if (ctx->remote_rank < 0) {
    MPI_Ibarrier(MPI_COMM_WORLD, &req1);
    do {
      ucp_worker_progress(ctx->ucp_worker);
      MPI_Test(&req1, &flag, MPI_STATUS_IGNORE);
    } while (!flag);
    MPI_Ibarrier(ctx->inter_comm, &req2);
    do {
      ucp_worker_progress(ctx->ucp_worker);
      MPI_Test(&req2, &flag, MPI_STATUS_IGNORE);
    } while (!flag);
  } else {
    MPI_Ibarrier(ctx->inter_comm, &req2);
    do {
      ucp_worker_progress(ctx->ucp_worker);
      MPI_Test(&req2, &flag, MPI_STATUS_IGNORE);
    } while (!flag);
  }
}

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCFinalizeContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "Begin _CAPI_HPCFinalizeContext";
  ContextRef ctx = args[0];
  ucs_status_t status;
  for (ucp_mem_h &mem : ctx->register_mem) {
    status = ucp_mem_unmap(ctx->ucp_context, mem);
    if (status != UCS_OK) {
      LOG(FATAL) << "ucp_mem_unmap failed with " << ucs_status_string(status);
    }
  }
  std::vector<ucs_status_ptr_t> stats;
  for (ucp_ep_h &ep : ctx->remote_ep) {
    stats.push_back(ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH));
  }
  for (ucs_status_ptr_t &stat : stats) {
    if (UCS_PTR_IS_ERR(stat)) {
      LOG(FATAL) << "ucp ep close failed";
    }
    if (stat && UCS_PTR_IS_PTR(stat)) {
      do {
        ucp_worker_progress(ctx->ucp_worker);
        status = UCS_PTR_STATUS(stat);
      } while (status == UCS_INPROGRESS && stat);
    }
    if (stat) {
      ucp_request_free(stat);
    }
  }
  LOG(INFO) << "begin sync_all_proc";
  sync_all_proc(ctx);
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


}  // namespace context
}  // namespace hpc
}  // namespace dgl
