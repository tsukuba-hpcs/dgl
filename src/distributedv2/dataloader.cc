#include "dataloader.h"
#include "context.h"


#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"

namespace dgl {
namespace distributedv2 {

using namespace dgl::runtime;

NodeDataLoader::NodeDataLoader(int rank, int size, Communicator *comm, node_dataloader_arg_t &&arg)
: ServiceManager(rank, size, comm) {
  
}

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2CreateNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  LOG(INFO) << "_CAPI_DistV2CreateNodeDataLoader is called";
  ContextRef ctx = args[0];
  int num_layers = args[1];
  int num_nodes = args[2];
  node_dataloader_arg_t arg = {
    .num_nodes = num_nodes,
    .num_layers = num_layers,
  };
  std::shared_ptr<NodeDataLoader> loader(new NodeDataLoader(ctx->rank, ctx->size, &ctx->comm, std::move(arg)));
  *rv = loader;
});

DGL_REGISTER_GLOBAL("distributedv2._CAPI_DistV2EnqueueToNodeDataLoader")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NodeDataLoaderRef loader = args[0];
  List<Value> seeds = args[1];
  int batch_size = args[2];
  LOG(INFO) << "seeds.size() = " << seeds.size();
});

}
}