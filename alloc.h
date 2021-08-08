#pragma once

#include <stdlib.h>

#include <atomics.h>
#include <utils.h>
#include <worker.h>

struct alignas(CACHELINE_SIZE) malloc_info_t {
  volatile char* head;
  volatile char* tail;
  uint64_t max_alloc_size;
};

// Should only be called on one worker
// INLINE void init_malloc(team_t* team, void* malloc_storage,
//                        uint64_t num_cachelines_per_sm)
//{
//  for (int i = 0; i < nsmid(); ++i) {
//    team->malloc_info[i].head = team->malloc_info[i].tail =
//        (char*)malloc_storage + CACHELINE_SIZE * num_cachelines_per_sm * i;
//  }
//  team->max_alloc_size = num_cachelines_per_sm * CACHELINE_SIZE;
//}

INLINE void* fast_malloc(worker_t* worker, uint64_t size)
{
  volatile char** head = &worker->team->malloc_info[smid()].head;
  volatile char** tail = &worker->team->malloc_info[smid()].tail;
  //uint64_t mas = worker->team->max_alloc_size;
  uint64_t mas = worker->team->malloc_info[smid()].max_alloc_size;

  // TODO this may be a race condition here
  if (*tail - *head >= mas) {
    printf("(%d) error, out of memory: %p, %p, %llu\n", smid(), *tail, *head, mas);
    asm("trap;");
    return nullptr;
  }
  else {
    return (void*)atomic_add((unsigned long long*)tail,
                             (unsigned long long)size);
  }
}
