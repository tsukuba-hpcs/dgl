/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/shard.h
 * \brief headers for HPC shard.
 */

#ifndef DGL_HPC_SHARD_H_
#define DGL_HPC_SHARD_H_

#include <dgl/runtime/c_runtime_api.h>

#include <dgl/runtime/object.h>
#include <unordered_map>
#include <string>
#include <vector>

namespace dgl {
namespace hpc {
namespace shard {


struct Shard : public runtime::Object {
  std::unordered_map<std::string, int> name2id;
  std::vector<runtime::NDArray> tensor;

  static constexpr const char* _type_key = "hpc.Shard";
  DGL_DECLARE_OBJECT_TYPE_INFO(Shard, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ShardRef, Shard);

struct TensorMetaData {
  DGLType dtype;
  int ndim;
  std::vector<int64_t> shape;
  std::vector<void*> data;
};

struct ShardClient : public runtime::Object {
  std::unordered_map<std::string, int> name2id;
  std::vector<TensorMetaData> metadata;

  static constexpr const char* _type_key = "hpc.ShardClient";
  DGL_DECLARE_OBJECT_TYPE_INFO(ShardClient, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ShardClientRef, ShardClient);

}  // namespace shard
}  // namespace hpc
}  // namespace dgl

#endif  // DGL_HPC_SHARD_H_
