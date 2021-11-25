/*!
 *  Copyright (c) 2021 by Contributors
 * \file distributedv2/context.h
 * \brief headers for distv2 context.
 */

#ifndef DGL_DISTV2_CONTEXT_H_
#define DGL_DISTV2_CONTEXT_H_

#include <ucp/api/ucp.h>
#include <dgl/runtime/object.h>

#include <vector>
#include "comm.h"
#include "service.h"


namespace dgl {
namespace distributedv2 {

struct Context: public runtime::Object {
  Communicator comm;
  ServiceManager manager;
  static constexpr const char* _type_key = "distributedv2.Context";
  Context(int rank, int size)
  : comm(rank, size)
  , manager(rank, size, &comm) {};
  DGL_DECLARE_OBJECT_TYPE_INFO(Context, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ContextRef, Context);

}  // namespace distributedv2
}  // namespace dgl


#endif  // DGL_DISTV2_CONTEXT_H_