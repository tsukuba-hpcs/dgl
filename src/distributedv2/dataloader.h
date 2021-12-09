/*!
 *  Copyright (c) 2021 by Contributors
 * \file distributedv2/dataloader.h
 * \brief headers for distv2 dataloader.
 */

#ifndef DGL_DISTV2_DATALOADER_H_
#define DGL_DISTV2_DATALOADER_H_

#include "service.h"
#include "neighbor.h"
#include <dmlc/concurrentqueue.h>
#include <vector>

namespace dgl {
namespace distributedv2 {

struct node_dataloader_arg_t {
  uint64_t num_nodes;
  uint16_t num_layers;
};

class NodeDataLoader: public ServiceManager, public runtime::Object {
  dmlc::moodycamel::ConcurrentQueue<std::vector<int>> input;
  dmlc::moodycamel::ConcurrentQueue<std::vector<int>> output;
public:
  static constexpr const char* _type_key = "distributedv2.NodeDataLoader";
  NodeDataLoader(int rank, int size, Communicator *comm, node_dataloader_arg_t &&arg);
  DGL_DECLARE_OBJECT_TYPE_INFO(NodeDataLoader, runtime::Object);
};

DGL_DEFINE_OBJECT_REF(NodeDataLoaderRef, NodeDataLoader);

}
}

#endif