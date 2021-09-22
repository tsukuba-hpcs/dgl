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
#include <vector>
#include <string>

#include "../c_api_common.h"



namespace dgl {
namespace hpc {

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

//////////////////////////// Manager ////////////////////////////

static void server_conn_handle_cb(ucp_conn_request_h conn_request, void *arg) {
  Context* ctx = reinterpret_cast<Context *>(arg);
  LOG(INFO) << "rank=" << ctx->rank << ": server_conn_handle_cb is called";
}

static inline void create_listener(const ContextRef &ctx, ucp_listener_h *ucp_listener) {
  ucs_status_t status;
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
  status = ucp_listener_create(ctx->ucp_worker, &params, ucp_listener);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_listener_create failed";
  }
}

static inline void get_sockaddr(struct sockaddr_in *sock, const ucp_listener_h &ucp_listener) {
  ucs_status_t status;
  ucp_listener_attr_t attr = {
    .field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR,
  };
  status = ucp_listener_query(ucp_listener, &attr);
  if (status != UCS_OK) {
    LOG(FATAL) << "ucp_listener_query failed";
  }
  std::memcpy(sock, &attr.sockaddr, sizeof(sockaddr_in));
}

static inline void gather_sockaddr(struct sockaddr_in *socks, const struct sockaddr_in &sock) {
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allgather(&sock, sizeof(sock), MPI_BYTE, socks, sizeof(sock), MPI_BYTE, MPI_COMM_WORLD);
}

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

static inline void bcast_manager_context(ContextRef ctx) {
  MPI_Barrier(ctx->inter_comm);
  MPI_Bcast(&ctx->rank, 1, MPI_INT32_T, MPI_ROOT, ctx->inter_comm);
  MPI_Bcast(&ctx->size, 1, MPI_INT32_T, MPI_ROOT, ctx->inter_comm);
}

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCManagerLaunchWorker")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ucp_listener_h ucp_listener;
  ContextRef ctx = args[0];
  int32_t num_workers = args[1];
  std::vector<std::string> worker_args;
  for (int i = 2; i < args.num_args; i++) {
    const std::string arg = args[i];
    worker_args.push_back(arg);
  }
  struct sockaddr_in sock;
  std::vector<struct sockaddr_in> socks(ctx->size);
  create_listener(ctx, &ucp_listener);
  get_sockaddr(&sock, ucp_listener);
  {
    char buffer[100];
    LOG(INFO) << "rank=" << ctx->rank << " listen "
    << inet_ntop(AF_INET, &sock.sin_addr, reinterpret_cast<char *>(&buffer), 100)
    << ":" << htons(sock.sin_port) << " ....";
  }
  gather_sockaddr(&socks[0], sock);
  spawn_worker(&ctx->inter_comm, num_workers, worker_args);
  ctx->remote_rank = -1;
  ctx->remote_size = num_workers;
  bcast_manager_context(ctx);
  ucp_listener_destroy(ucp_listener);
});

//////////////////////////// Worker ////////////////////////////

static inline void recv_manager_context(ContextRef ctx) {
  MPI_Barrier(ctx->inter_comm);
  MPI_Bcast(&ctx->remote_rank, 1, MPI_INT32_T, 0, ctx->inter_comm);
  MPI_Bcast(&ctx->remote_size, 1, MPI_INT32_T, 0, ctx->inter_comm);
  LOG(INFO) << "remote_rank=" << ctx->remote_rank << " "
            << "remote_size=" << ctx->remote_size;
}

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCWorkerConnect")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  MPI_Comm_get_parent(&ctx->inter_comm);
  recv_manager_context(ctx);
});

}  // namespace hpc
}  // namespace dgl
