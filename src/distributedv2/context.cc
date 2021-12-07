#include "context.h"

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"

namespace dgl {
namespace distributedv2 {

using namespace dgl::runtime;

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "_CAPI_DistV2CreateContext is called";
  int rank = args[0];
  int size = args[1];
  std::shared_ptr<Context> ctx(new Context(rank, size));
  *rv = ctx;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2GetWorkerAddr")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  List<Value> ret;
  auto p = ctx->comm.get_workeraddr();
  ret.push_back(Value(MakeValue(p.first)));
  ret.push_back(Value(MakeValue(p.second)));
  *rv = ret;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateEndpoints")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ContextRef ctx = args[0];
  std::string addrs = args[1];
  ctx->comm.create_endpoints(addrs);
});

} // namespace distributedv2
} // namespace dgl