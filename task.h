#pragma once

#include <stdint.h>

#include <alloc.h>
#include <atomics.h>
#include <utils.h>
#include <worker.h>

struct task_t;

typedef void (*task_func_t)(worker_t*, task_t*);

struct alignas(CACHELINE_SIZE) task_t {
  task_func_t func;
  task_t* parent;
  volatile uint32_t num_children;
  // 'storage' must be 8-byte aligned
  // TODO think of better way to handle this
  volatile uint32_t num_shareds;
  uint32_t num_privates;
  uint32_t pad;
  char storage[CACHELINE_SIZE - sizeof(task_func_t) - sizeof(task_t*) -
               sizeof(uint32_t) * 4];
};

static_assert(sizeof(task_t) == CACHELINE_SIZE,
              "error: task size not equal to cacheline size");

INLINE task_t* alloc_task(worker_t* worker, task_func_t func,
                          uint32_t nprivates, void** privates)
{
  // tasks are just the length of a cacheline
  task_t* task = (task_t*)fast_malloc(worker, CACHELINE_SIZE);
  task_t* parent_task = worker->current_task;

  task->func = func;
  task->num_children = 0;
  task->num_shareds = 0;
  task->num_privates = nprivates;
  task->parent = parent_task;

  void** ptr = (void**)task->storage;
  for (int i = 0; i < nprivates; ++i) *ptr++ = privates[i];

  return task;
}

INLINE void execute_task(worker_t* worker, task_t* task)
{
  worker->current_task = task;
  task->func(worker, task);
}

INLINE void finish_task(worker_t* worker, task_t* task)
{
  if (gpu_utils::thread_id() == 0) {
    worker->current_task = nullptr;
    atomic_sub(&worker->team->num_barrier_tasks, 1u);
    // TODO this check seems a bit pointless because only 1 task should ever
    // not have a parent
    if (task->parent != nullptr) {
      atomic_sub(&task->parent->num_children, 1u);
    }
    // TODO is this needed?
    __threadfence();
  }
}

// ---------
// User code
// ---------

INLINE void* create_shared(task_t* task)
{
  // TODO probably not thread safe
  return ((void**)task->storage + task->num_privates + task->num_shareds++);
}

INLINE void* get_private(task_t* task, uint32_t i)
{
  return ((void**)task->storage)[i];
}
