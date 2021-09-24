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

DGL_REGISTER_GLOBAL("hpc.shard._CAPI_HPCRegisterTensor")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  ShardRef shard = args[0];
  std::string name = args[1];
  NDArray tensor = args[2];
  if (shard->name2id.count(name)) {
    LOG(FATAL) << name << " is already exists in Shard";
  }
  int id = shard->name2id.size();
  if (static_cast<int>(shard->tensor.size()) != id) {
    LOG(FATAL) << "number of tensor is not equal to number of name";
  }
  shard->name2id[name] = id;
  shard->tensor.push_back(tensor);
  *rv = id;
});

}  // namespace shard
}  // namespace hpc
}  // namespace dgl
