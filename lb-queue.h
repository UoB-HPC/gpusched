#pragma once

#include <stdlib.h>

#include <task.h>
#include <ticket-lock.h>
#include <worker.h>

#ifndef QUEUE_SIZE
#define QUEUE_SIZE 256
#endif

#define QUEUE_MASK (QUEUE_SIZE - 1);

static_assert(QUEUE_SIZE > 0 && ((QUEUE_SIZE & (QUEUE_SIZE - 1)) == 0),
              "queue size must be power of two");

struct alignas(CACHELINE_SIZE) queue_t {
  task_t** storage;
  volatile uint32_t head;
  volatile uint32_t tail;
  volatile uint32_t ntasks;
  // seems to be better to have the lock in its own cache line and the rest of
  // the elements in another cacheline
  alignas(CACHELINE_SIZE) lock_t lock;
};

INLINE void init_queue(queue_t* queue)
{
  // Does not happen often, okay to do (slow) device-side malloc
  queue->storage = (task_t**)aligned_malloc(sizeof(task_t*) * QUEUE_SIZE, 128);
  queue->head = 0;
  queue->tail = 0;
  queue->ntasks = 0;
  init_lock(&queue->lock);
}

INLINE void reset_queue(queue_t* queue)
{
  queue->head = 0;
  queue->tail = 0;
  queue->ntasks = 0;
  init_lock(&queue->lock);
}

INLINE void fini_queue(queue_t* queue) { aligned_free(queue->storage); }

INLINE bool enqueue_back(worker_t* worker, task_t* task)
{
  queue_t* queue = worker->queue;
  if (queue->ntasks == QUEUE_SIZE) return false;
  lock(&queue->lock);
  if (queue->ntasks == QUEUE_SIZE) {
    unlock(&queue->lock);
    return false;
  }
  // TODO avoid unnecessary reloading wrt. volatile
  uint32_t tail = queue->tail;
  queue->storage[tail] = task;
  tail = (tail + 1) & QUEUE_MASK;
  queue->tail = tail;
  queue->ntasks++;
  unlock(&queue->lock);
  return true;
}

INLINE task_t* dequeue_back(worker_t* worker)
{
  queue_t* queue = worker->queue;
  if (queue->ntasks == 0) return nullptr;
  lock(&queue->lock);
  if (queue->ntasks == 0) {
    unlock(&queue->lock);
    return nullptr;
  }
  // TODO avoid unnecessary reloading wrt. volatile
  uint32_t tail = queue->tail;
  tail = (tail - 1) & QUEUE_MASK;
  task_t* ret = queue->storage[tail];
  queue->tail = tail;
  queue->ntasks--;
  unlock(&queue->lock);
  return ret;
}

INLINE task_t* dequeue_front(worker_t* worker)
{
  queue_t* queue = worker->queue;
  if (queue->ntasks == 0) return nullptr;
  lock(&queue->lock);
  if (queue->ntasks == 0) {
    unlock(&queue->lock);
    return nullptr;
  }
  // TODO avoid unnecessary reloading wrt. volatile
  uint32_t head = queue->head;
  task_t* ret = queue->storage[head];
  head = (head + 1) & QUEUE_MASK;
  queue->head = head;
  queue->ntasks--;
  unlock(&queue->lock);
  return ret;
}
