#pragma once

#include <stdlib.h>

#include <gpu-utils.h>
#include <utils.h>

struct worker_t;
struct task_t;
struct queue_t;
struct malloc_info_t;

struct alignas(CACHELINE_SIZE) team_t {
  worker_t* workers;
  malloc_info_t* malloc_info;
  uint32_t num_blocks;
  //uint64_t max_alloc_size;

  // TODO we access this variable so much in parallel, it should really be in
  // its own cacheline
  alignas(CACHELINE_SIZE) volatile uint32_t num_barrier_tasks;
  alignas(CACHELINE_SIZE) volatile uint32_t barrier_val;
  alignas(CACHELINE_SIZE) volatile uint32_t barrier_counter;
};

struct alignas(CACHELINE_SIZE) worker_t {
  queue_t* queue;
  team_t* team;
  task_t* current_task;
  uint32_t id;
  uint32_t rand_x;
  uint32_t rand_a;
#ifdef SCHED_MEMORISE
  uint32_t last_stolen;
#endif
};

// Implementation of linear congruential engine taken from LLVM
__device__ static const unsigned primes[] = {
    0x9e3779b1, 0xffe6cc59, 0x2109f6dd, 0x43977ab5, 0xba5703f5, 0xb495a877,
    0xe1626741, 0x79695e6b, 0xbc98c09f, 0xd5bee2b3, 0x287488f9, 0x3af18231,
    0x9677cd4d, 0xbe3a6929, 0xadc6a877, 0xdcf0674b, 0xbe4d6fe9, 0x5f15e201,
    0x99afc3fd, 0xf3f16801, 0xe222cfff, 0x24ba5fdb, 0x0620452d, 0x79f149e3,
    0xc8b93f49, 0x972702cd, 0xb07dd827, 0x6c97d5ed, 0x085a3d61, 0x46eb5ea7,
    0x3d9910ed, 0x2e687b5b, 0x29609227, 0x6eb081f1, 0x0954c4e1, 0x9d114db9,
    0x542acfa9, 0xb3e6bd7b, 0x0742d917, 0xe9f3ffa7, 0x54581edb, 0xf2480f45,
    0x0bb9288f, 0xef1affc7, 0x85fa0ca7, 0x3ccc14db, 0xe6baf34b, 0x343377f7,
    0x5ca19031, 0xe6d9293b, 0xf0a9f391, 0x5d2e980b, 0xfc411073, 0xc3749363,
    0xb892d829, 0x3549366b, 0x629750ad, 0xb98294e5, 0x892d9483, 0xc235baf3,
    0x3d2402a3, 0x6bdef3c9, 0xbec333cd, 0x40c9520f};

INLINE unsigned short get_random(worker_t* worker)
{
  unsigned x = worker->rand_x;
  unsigned short r = x >> 16;
  worker->rand_x = x * worker->rand_a + 1;
  return r;
}

INLINE void init_random(worker_t* worker)
{
  unsigned seed = worker->id;
  worker->rand_a = primes[seed % (sizeof(primes) / sizeof(primes[0]))];
  worker->rand_x = (seed + 1) * worker->rand_a + 1;
}

