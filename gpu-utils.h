#pragma once

#include <cstdint>

#include <utils.h>

// By default each thread block has 4 workers, each with 32 threads
#ifndef NTHREADS
#define NTHREADS 32
#endif

#ifndef NWORKERS
#define NWORKERS 4
#endif

#if NTHREADS > 32 && NWORKERS > 1
#error only one worker allowed when threads g.t. 32
#endif

namespace gpu_utils {
static const int warp_size = 32;
static const int num_threads = NTHREADS;

extern __shared__ int32_t __shmem_all[];

#if NWORKERS > 1
INLINE int thread_id() { return threadIdx.x % warp_size; }
INLINE int worker_id() { return threadIdx.x / warp_size; }
INLINE int global_worker_id() { return blockIdx.x * NWORKERS + worker_id(); }
#else
INLINE int thread_id() { return threadIdx.x; }
INLINE int worker_id() { return 0; }
INLINE int global_worker_id() { return blockIdx.x; }
INLINE void* shmem() { return (void*)__shmem_all; } 
#endif


#if NTHREADS > 32
INLINE void sync_worker() { __syncthreads(); }
template <class T>
INLINE T broadcast(T val, int root)
{
  __shared__ T broadcast_shmem[1];
  if (thread_id() == root) broadcast_shmem[0] = val;
  __syncthreads();
  return broadcast_shmem[0];
}
#else
INLINE void sync_worker() { __syncwarp(); }
template <class T>
INLINE T broadcast(T val, int root)
{
  return __shfl_sync(0xffffffff, val, root);
}
#endif
}  // namespace gpu_utils
