/*!
 *  Copyright (c) 2021 by Contributors
 * \file ucx_communicator.h
 * \brief UCXCommunicator for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_UCX_COMMUNICATOR_H_
#define DGL_RPC_NETWORK_UCX_COMMUNICATOR_H_

#include <ucp/api/ucp.h>
#include <dmlc/concurrentqueue.h>

#include <string>
#include <thread>
#include <unordered_map>
#include <memory>
#include <vector>

#include "common.h"
#include "communicator.h"

namespace dgl {
namespace network {
/*!
 * \breif Networking address
 */
struct IPAddr {
  std::string ip;
  int port;
};

struct UCXStreamBuffer {
  int64_t data_size;

  char size_buffer[sizeof(int64_t)];
  int64_t size_offset;
  char* data_buffer;
  int64_t data_offset;

  UCXStreamBuffer() :
    data_size(0),
    size_offset(0),
    data_buffer(NULL),
    data_offset(0) {}
};

class UCXSender : public Sender {
 public:
  /*!
   * \brief Sender constructor
   * \param queue_size size of message queue 
   */
  explicit UCXSender(int64_t queue_size);

  /*!
   * \brief Add receiver's address and ID to the sender's namebook
   * \param addr Networking address, e.g., 'ucx://127.0.0.1:50091'
   * \param id receiver's ID
   *
   * AddReceiver() is not thread-safe and only one thread can invoke this API.
   */
  void AddReceiver(const char* addr, int recv_id);

  /*!
   * \brief Connect with all the Receivers
   * \return True for success and False for fail
   *
   * Connect() is not thread-safe and only one thread can invoke this API.
   */
  bool Connect();

  /*!
   * \brief Send data to specified Receiver. Actually pushing message to message queue.
   * \param msg data message
   * \param recv_id receiver's ID
   * \return Status code
   *
   * (1) The send is non-blocking. There is no guarantee that the message has been 
   *     physically sent out when the function returns.
   * (2) The communicator will assume the responsibility of the given message.
   * (3) The API is multi-thread safe.
   * (4) Messages sent to the same receiver are guaranteed to be received in the same order. 
   *     There is no guarantee for messages sent to different receivers.
   */
  STATUS Send(Message msg, int recv_id);

  /*!
   * \brief Finalize SocketSender
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

  /*!
   * \brief Communicator type: 'socket'
   */
  inline std::string Type() const { return std::string("ucx"); }

 private:
  /*!
   * \brief ucp endpoint for each connection of receiver
   */ 
  std::unordered_map<int /* receiver ID */, ucp_ep_h> eps_;

  /*!
   * \brief receivers' address
   */ 
  std::unordered_map<int /* receiver ID */, IPAddr> receiver_addrs_;

  /*!
   * \brief ucp context
   */
  ucp_context_h context_;

  /*!
   * \brief ucp worker
   */
  ucp_worker_h worker_;

  /*!
   * \brief progress queue
   */
  dmlc::moodycamel::ConcurrentQueue<ucs_status_ptr_t> prog_queue_;

  /*!
   * \brief progress thread
   */
  std::shared_ptr<std::thread> prog_thread_;

  /*!
   * \brief Send-loop to run ucp_worker_progress
   * \param self self pointer
   * 
   * Note that, the SendLoop will finish its loop-job and exit thread
   * when the main thread push NULL to prog_queue_
   */
  static void SendLoop(UCXSender *self);
};

class UCXReceiver : public Receiver {
 public:
  /*!
   * \brief Receiver constructor
   * \param queue_size size of message queue.
   */
  explicit UCXReceiver(int64_t queue_size);

  /*!
   * \brief Wait for all the Senders to connect
   * \param addr Networking address, e.g., 'ucx://127.0.0.1:50091'
   * \param num_sender total number of Senders
   * \return True for success and False for fail
   *
   * Wait() is not thread-safe and only one thread can invoke this API.
   */
  bool Wait(const char* addr, int num_sender);

  /*!
   * \brief Recv data from Sender.
   * \param msg pointer of data message
   * \param send_id which sender current msg comes from
   * \return Status code
   *
   * (1) The Recv() API is blocking, which will not 
   *     return until getting data from message queue.
   * (2) The Recv() API is thread-safe.
   * (3) Memory allocated by communicator but will not own it after the function returns.
   */
  STATUS Recv(Message* msg, int* send_id);

  /*!
   * \brief Recv data from a specified Sender. Actually removing data from msg_queue.
   * \param msg pointer of data message
   * \param send_id sender's ID
   * \return Status code
   *
   * (1) The RecvFrom() API is blocking, which will not 
   *     return until getting data from message queue.
   * (2) The RecvFrom() API is thread-safe.
   * (3) Memory allocated by communicator but will not own it after the function returns.
   */
  STATUS RecvFrom(Message* msg, int send_id);

  /*!
   * \brief Finalize SocketSender
   *
   * Finalize() is not thread-safe and only one thread can invoke this API.
   */
  void Finalize();

 private:
  /*!
   * \brief Queue size
   */ 
  int64_t queue_size;
  /*!
   * \brief ucp endpoint for each connection of receiver
   */ 
  std::vector<int /* sender ID */, ucp_ep_h> eps_;

  /*!
   * \brief ucp context
   */
  ucp_context_h context_;

  /*!
   * \brief ucp worker
   */
  ucp_worker_h worker_;

  /*!
   * \brief ucp listener
   */
  ucp_listener_h listener_;

  /*!
   * \brief notify queue for Recv()
   */
  dmlc::moodycamel::ConcurrentQueue<int> notify_queue_;

  /*!
   * \brief receive queue for each sender.
   */
  std::vector<dmlc::moodycamel::ConcurrentQueue<Message>> recv_queues_;

  /*!
   * \brief progress thread
   */
  std::shared_ptr<std::thread> prog_thread_;

  /*!
   * \brief RecvMsgLoop to receive msgs from stream
   * \param self self pointer
   * \param length buffer length
   * \param data data pointer
   * 
   */
  static std::vector<Message>
    RecvMsgLoop(UCXStreamBuffer *buf, size_t length, ucs_status_ptr_t data);

  /*!
   * \brief Recv-loop to run ucp_worker_progress
   * \param self self pointer
   * 
   * Note that, the RecvLoop will finish its loop-job and exit thread
   * when the main thread push NULL to prog_queue_
   */
  static void RecvLoop(UCXReceiver *self);

  /*!
   * \brief callback for endpoint connection.
   * \param arg self pointer
   * 
   *  this is called when client creates endpoint for this receiver.
   */
  static void ConnHandlerCallback(ucp_conn_request_h conn_request, void *arg);
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_UCX_COMMUNICATOR_H_
