/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/shard.h
 * \brief headers for HPC shard.
 */

#ifndef DGL_HPC_SHARD_H_
#define DGL_HPC_SHARD_H_

#include <dgl/runtime/object.h>
#include <unordered_map>
#include <string>

namespace dgl {
namespace hpc {
namespace shard {


struct Shard : public runtime::Object {
  int32_t rank;
  static constexpr const char* _type_key = "hpc.Shard";
  DGL_DECLARE_OBJECT_TYPE_INFO(Shard, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ShardRef, Shard);

}  // namespace shard
}  // namespace hpc
}  // namespace dgl

#endif  // DGL_HPC_SHARD_H_
