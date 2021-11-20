/*!
 *  Copyright (c) 2021 by Contributors
 * \file ucx/context.h
 * \brief headers for UCX context.
 */

#ifndef DGL_UCX_CONTEXT_H_
#define DGL_UCX_CONTEXT_H_

#include <ucp/api/ucp.h>
#include <dgl/runtime/object.h>

#include <vector>
#include "service.h"


namespace dgl {
namespace ucx {
namespace context {

struct Context: public runtime::Object {
  int32_t rank;
  int32_t size;
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  ucp_address_t* addr;
  size_t addrlen;
  std::vector<ucp_ep_h> eps;
  service::ServiceManager manager;
  static constexpr const char* _type_key = "ucx.Context";
  DGL_DECLARE_OBJECT_TYPE_INFO(Context, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(ContextRef, Context);


}  // namespace context
}  // namespace ucx
}  // namespace dgl

#endif  // DGL_UCX_CONTEXT_H_