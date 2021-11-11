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
  *rv = ctx;
});

DGL_REGISTER_GLOBAL("distributedv2.context._CAPI_UCXFinalizeContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  LOG(INFO) << "_CAPI_UCXFinalizeContext is called";
  ucp_cleanup(ctx->ucp_context);
});

} // namespace context
} // namespace dgl
} // namespace ucx