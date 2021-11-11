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
  std::shared_ptr<Context> ctx(new Context);
  *rv = ctx;
});

} // namespace context
} // namespace dgl
} // namespace ucx