#include "context.h"

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"

namespace dgl {
namespace ucx {
namespace context {

using namespace dgl::runtime;

DGL_REGISTER_GLOBAL("distributedv2.context._CAPI_UCXCreateContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "_CAPI_UCXCreateContext is called";
  int rank = args[0];
  int size = args[1];
  std::shared_ptr<Context> ctx(new Context);
  ctx->rank = rank;
  ctx->size = size;
  ucs_status_t status;
  ucp_params_t params = {
    .field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS,
    .features = UCP_FEATURE_STREAM | UCP_FEATURE_RMA,
    .estimated_num_eps = static_cast<size_t>(size),
  };
  if ((status = ucp_init(&params, NULL, &ctx->ucp_context)) != UCS_OK) {
    LOG(FATAL) << "ucp_init error: " << ucs_status_string(status);
  }
  ucp_worker_params_t wparams = {
    .field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
    .thread_mode = UCS_THREAD_MODE_SERIALIZED,
  };
  if ((status = ucp_worker_create(ctx->ucp_context, &wparams, &ctx->ucp_worker)) != UCS_OK) {
    LOG(FATAL) << "ucp_worker_create error: " << ucs_status_string(status);
  }
  if ((status = ucp_worker_get_address(ctx->ucp_worker, &ctx->addr, &ctx->addrlen)) != UCS_OK) {
    LOG(FATAL) << "ucp_worker_get_address error: " << ucs_status_string(status);
  }
  *rv = ctx;
});

DGL_REGISTER_GLOBAL("distributedv2.context._CAPI_UCXGetWorkerAddr")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  *rv = ctx->addr;
});

DGL_REGISTER_GLOBAL("distributedv2.context._CAPI_UCXGetWorkerAddrlen")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  int len = static_cast<int>(ctx->addrlen);
  *rv = len;
});

DGL_REGISTER_GLOBAL("distributedv2.context._CAPI_UCXCreateEndpoints")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  std::string addrs = args[1];
  CHECK(addrs.length() == ctx->size * ctx->addrlen);
  ucs_status_t status;
  ctx->eps.resize(ctx->size);
  for (int rank = 0; rank != ctx->size; rank++) {
    if (rank == ctx->rank) continue;
    const ucp_address_t* addr =
      reinterpret_cast<const ucp_address_t *>(&addrs[ctx->addrlen * rank]);
    ucp_ep_params_t params = {
      .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS,
      .address = addr,
    };
    if ((status = ucp_ep_create(ctx->ucp_worker, &params, &ctx->eps[rank])) != UCS_OK) {
      LOG(FATAL) << "rank=" << rank
        <<"ucp_worker_get_address error: " << ucs_status_string(status);
    }
  }
});

DGL_REGISTER_GLOBAL("distributedv2.context._CAPI_UCXFinalizeContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  LOG(INFO) << "_CAPI_UCXFinalizeContext is called";
  for (int rank = 0; rank != ctx->size; rank++) {
    if (rank == ctx->rank) continue;
    ucp_ep_destroy(ctx->eps[rank]);
  }
  ucp_worker_release_address(ctx->ucp_worker, ctx->addr);
  ucp_worker_destroy(ctx->ucp_worker);
  ucp_cleanup(ctx->ucp_context);
});

} // namespace context
} // namespace dgl
} // namespace ucx