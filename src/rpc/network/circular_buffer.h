/*!
 *  Copyright (c) 2021 by Contributors
 * \file circular_buffer.h
 * \brief Circular Buffer for DGL distributed training.
 */
#ifndef DGL_RPC_NETWORK_CIRCULAR_BUFFER_H_
#define DGL_RPC_NETWORK_CIRCULAR_BUFFER_H_

namespace dgl {
namespace network {

#include <vector>
#include <atomic>
#include <utility>
#include <memory>

template <typename T>
class CircularBuffer;

/*!
 * \brief Producer which can push() on CircularBuffer.
 *
 * CircularBufferProducer can push() on CircularBuffer. 
 * Note:
 * 1. NOT thread-safe.
 * 2. Only one instance can be created at a time (single-producer).
 * 3. Before push(), user have to check if queue is filled, by fill().
 */ 
template <typename T>
class CircularBufferProducer {
 public:
  /*!
   * \brief CircularBufferProducer constructor
   * \param buffer shared pointer for CircularBuffer.
   */
  explicit CircularBufferProducer(std::shared_ptr<CircularBuffer<T>> buffer) {
    while (buffer->has_producer.exchange(true, std::memory_order_acquire)) {
    }
    buf = buffer;
  }

  /*!
   * \brief CircularBufferProducer deconstructor
   */
  ~CircularBufferProducer() {
    buf->has_producer.store(false, std::memory_order_release);
  }

  /*!
   * \brief Check if queue is filled.
   * \return If queue is filled, returns true.
   */
  bool fill() {
    return size()+1 == buf->size;
  }

  /*!
   * \brief Queue size.
   * \return number of items which is stored in queue.
   */
  int64_t size() {
    int64_t head, tail;
    head = buf->head.load(std::memory_order_acquire);
    tail = buf->tail.load();
    if (tail >= head) {
      return tail-head;
    }
    return tail+buf->size-head;
  }

  /*!
   * \brief Push item.
   * \param item 
   */
  void push(T item) {
    int64_t tail;
    tail = buf->tail.load();
    buf->data[tail] = std::move(item);
    buf->tail.store((tail+1)%buf->size, std::memory_order_release);
  }

 private:
  /*!
   * \brief shared pointer for CircularBuffer.
   */
  std::shared_ptr<CircularBuffer<T>> buf;
};

/*!
 * \brief Consumer which can pop() and front() on CircularBuffer.
 *
 * CircularBufferConsumer can pop() and front() on CircularBuffer. 
 * Note:
 * (1) NOT thread-safe.
 * (2) Only one instance can be created at a time (single-consumer).
 * (3) Before pop(), user have to check if queue is empty, by empty().
 * (4) If queue is empty, front() returns NULL.
 */ 
template <typename T>
class CircularBufferConsumer {
 public:
  /*!
   * \brief CircularBufferConsumer constructor
   * \param buffer shared pointer for CircularBuffer.
   */
  explicit CircularBufferConsumer(std::shared_ptr<CircularBuffer<T>> buffer) {
    while (buffer->has_consumer.exchange(true, std::memory_order_acquire)) {
    }
    buf = buffer;
  }

  /*!
   * \brief CircularBufferConsumer deconstructor
   */
  ~CircularBufferConsumer() {
    buf->has_consumer.store(false, std::memory_order_release);
  }

  /*!
   * \brief Check if queue is empty.
   * \return If queue is empty, returns true.
   */
  bool empty() {
    return size() == 0;
  }

  /*!
   * \brief Queue size.
   * \return number of items which is stored in queue.
   */
  int64_t size() {
    int64_t head, tail;
    head = buf->head.load();
    tail = buf->tail.load(std::memory_order_acquire);
    if (tail >= head) {
      return tail-head;
    }
    return tail+buf->size-head;
  }

  /*!
   * \brief get pointer of front item.
   * \return front item's pointer.
   * 
   * After called pop(), the pointer is undefined.
   * If queue is empty, the pointer is NULL.
   */
  T* front() {
    int64_t head, tail;
    head = buf->head.load();
    tail = buf->tail.load(std::memory_order_acquire);
    if (head == tail) {
      // empty
      return NULL;
    }
    return &buf->data[head];
  }

  /*!
   * \brief Pop item.
   * \return item.
   * 
   * If queue is empty, the item is undefined.
   */
  T pop() {
    int64_t head;
    head = buf->head.load();
    T ret = std::move(buf->data[head]);
    buf->head.store((head+1)%buf->size, std::memory_order_release);
    return ret;
  }

 private:
  /*!
   * \brief shared pointer for CircularBuffer.
   */
  std::shared_ptr<CircularBuffer<T>> buf;
};

/*!
 * \brief Lock-free Circular Buffer for single-consumer and single-producer.
 *
 * To push() on Circular Buffer, create CircularBufferProducer.
 * To pop() or front() on Circular Buffer, create CircularBufferConsumer.
 */ 
template <typename T>
class CircularBuffer {
 public:
  friend CircularBufferConsumer<T>;
  friend CircularBufferProducer<T>;

  explicit CircularBuffer(int64_t queue_size) :
    size(queue_size+1),
    data(queue_size+1),
    head(0),
    tail(0),
    has_producer(false),
    has_consumer(false) {
  }

 private:
  /*!
   * \brief Queue size
   */
  const int64_t size;

  /*!
   * \brief Buffer
   */
  std::vector<T> data;

  /*!
   * \brief head index.
   */
  std::atomic<int64_t> head;

  /*!
   * \brief tail index.
   */
  std::atomic<int64_t> tail;

  /*!
   * \brief producer is exist or not.
   */
  std::atomic_bool has_producer;

  /*!
   * \brief consumer is exist or not.
   */
  std::atomic_bool has_consumer;
};

}  // namespace network
}  // namespace dgl

#endif  // DGL_RPC_NETWORK_CIRCULAR_BUFFER_H_
