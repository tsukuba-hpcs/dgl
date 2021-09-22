/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/hpc.cc
 * \brief Implementation of HPC module.
 */

#include "hpc.h"
#include <arpa/inet.h>

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include <mpi.h>
#include <memory>

#include "../c_api_common.h"



namespace dgl {
namespace hpc {

using namespace dgl::runtime;

//////////////////////////// C APIs ////////////////////////////

//////////////////////////// Callback ////////////////////////////

static void server_conn_handle_cb(ucp_conn_request_h conn_request, void *arg) {
  Context* ctx = reinterpret_cast<Context *>(arg);
  LOG(INFO) << "rank=" << ctx->rank << ": server_conn_handle_cb is called";
}

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
    LOG(FATAL) << "ucp_init failed";
  }
  ucp_worker_params_t worker_params = {
    .field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
    .thread_mode = UCS_THREAD_MODE_SINGLE,
  };
  status = ucp_worker_create(rst->ucp_context,
    &worker_params, &rst->ucp_worker);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_worker_create failed";
  }
  *rv = rst;
});

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCFinalizeContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
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

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCContextLaunchWorker")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ucs_status_t status;
  ucp_listener_h ucp_listener;
  ContextRef ctx = args[0];
  struct sockaddr_in listen_addr = {
    .sin_family = AF_INET,
    .sin_port = htons(0),
    .sin_addr = {
        .s_addr = INADDR_ANY,
    },
  };
  ucp_listener_params_t params = {
    .field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                  UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
    .sockaddr = {
      .addr = (const struct sockaddr*)&listen_addr,
      .addrlen = sizeof(listen_addr)
    },
    .conn_handler = {
      .cb = server_conn_handle_cb,
      .arg = ctx.sptr().get(),
    }
  };
  status = ucp_listener_create(ctx->ucp_worker, &params, &ucp_listener);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_listener_create failed";
  }
  ucp_listener_attr_t attr = {
    .field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR,
  };
  status = ucp_listener_query(ucp_listener, &attr);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_listener_query failed";
  }
  struct sockaddr_in sock;
  std::memcpy(&sock, &attr.sockaddr, sizeof(sockaddr_in));
  {
    char buffer[100];
    LOG(INFO) << "rank=" << ctx->rank << " listen "
    << inet_ntop(AF_INET, &sock.sin_addr, reinterpret_cast<char *>(&buffer), 100)
    << ":" << htons(sock.sin_port) << " ....";
  }
  ucp_listener_destroy(ucp_listener);
});

}  // namespace hpc
}  // namespace dgl
