/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/manager.h
 * \brief headers for HPC manager.
 */

#ifndef DGL_HPC_MANAGER_H_
#define DGL_HPC_MANAGER_H_

#include <unordered_map>
#include <string>
#include <vector>

#include "context.h"

namespace dgl {
namespace hpc {
namespace manager {

struct Shard : public runtime::Object {
  std::unordered_map<std::string, int> name2id;
  std::vector<runtime::NDArray> tensor;

  static constexpr const char* _type_key = "hpc.manager.Shard";
  DGL_DECLARE_OBJECT_TYPE_INFO(Shard, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ShardRef, Shard);

}  // namespace manager
}  // namespace hpc
}  // namespace dgl

#endif  // DGL_HPC_MANAGER_H_
