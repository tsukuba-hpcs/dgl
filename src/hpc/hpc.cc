/*!
 *  Copyright (c) 2021 by Contributors
 * \file hpc/hpc.cc
 * \brief Implementation of HPC module.
 */

#include <mpi.h>
#include <memory>
#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"
#include "hpc.h"

namespace dgl {
namespace hpc {

using namespace dgl::runtime;

//////////////////////////// C APIs ////////////////////////////


//////////////////////////// Context ////////////////////////////

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCCreateContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int flag;
  MPI_Initialized(&flag);
  if (flag) {
    LOG(FATAL) << "MPI is already Initialized";
  }
  MPI_Init(NULL, NULL);
  std::shared_ptr<Context> rst(new Context);
  MPI_Comm_rank(MPI_COMM_WORLD, &rst->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rst->size);
  *rv = rst;
});

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCFinalizeContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int flag;
  MPI_Initialized(&flag);
  if (!flag) {
    LOG(FATAL) << "MPI is not Initialized";
  }
  MPI_Finalize();
});

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCContextGetRank")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const ContextRef ctx = args[0];
  *rv = ctx->rank;
});

DGL_REGISTER_GLOBAL("hpc.context._CAPI_HPCContextGetSize")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const ContextRef ctx = args[0];
  *rv = ctx->size;
});

}  // namespace hpc
}  // namespace dgl
