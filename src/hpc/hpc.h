/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/hpc.h
 * \brief Common headers for HPC module.
 */

#ifndef DGL_HPC_HPC_H_
#define DGL_HPC_HPC_H_

#include <ucp/api/ucp.h>
#include <mpi.h>

#include <dgl/runtime/object.h>
#include <cstdint>

namespace dgl {
namespace hpc {

struct Context : public runtime::Object {
  int32_t rank;
  int32_t size;
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  MPI_Comm inter_comm;
  int32_t remote_rank; // for worker only
  int32_t remote_size;

  static constexpr const char* _type_key = "hpc.Context";
  DGL_DECLARE_OBJECT_TYPE_INFO(Context, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ContextRef, Context);

}  // namespace hpc
}  // namespace dgl
#endif  // DGL_HPC_HPC_H_
