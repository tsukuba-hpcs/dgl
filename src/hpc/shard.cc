/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/shard.cc
 * \brief Implementation of HPC shard.
 */

#include "shard.h"

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <memory>

#include "../c_api_common.h"

namespace dgl {
namespace hpc {
namespace shard {

using namespace dgl::runtime;

//////////////////////////// C APIs ////////////////////////////

//////////////////////////// Shard ////////////////////////////


DGL_REGISTER_GLOBAL("hpc.shard._CAPI_HPCCreateShard")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  std::shared_ptr<Shard> rst(new Shard);
  LOG(INFO) << "CreateShard called";
  *rv = rst;
});

}  // namespace shard
}  // namespace hpc
}  // namespace dgl
