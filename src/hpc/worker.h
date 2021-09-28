/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/client.h
 * \brief headers for HPC client.
 */

#ifndef DGL_HPC_WORKER_H_
#define DGL_HPC_WORKER_H_

#include <ucp/api/ucp.h>

#include <dgl/runtime/c_runtime_api.h>

#include <dgl/runtime/object.h>
#include <unordered_map>
#include <string>
#include <vector>

namespace dgl {
namespace hpc {
namespace worker {


struct TensorMetaData {
  DGLType dtype;
  int col_ndim;
  std::vector<int64_t> col_shape;
  std::vector<void*> data;
  std::vector<ucp_rkey_h> rkeys;
};

struct ShardClient : public runtime::Object {
  std::unordered_map<std::string, int> name2id;
  std::vector<TensorMetaData> metadata;

  static constexpr const char* _type_key = "hpc.ShardClient";
  DGL_DECLARE_OBJECT_TYPE_INFO(ShardClient, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ShardClientRef, ShardClient);

}  // namespace worker
}  // namespace hpc
}  // namespace dgl

#endif  // DGL_HPC_WORKER_H_
