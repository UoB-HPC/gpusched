#pragma once

#include <stdio.h>
#include <stdint.h>

#ifdef __CUDACC__
#define CACHELINE_SIZE 128
#define INLINE __device__ __forceinline__

#define CUDACHK(ans)                                                           \
  {                                                                            \
    sched_gpu_assert((ans), __FILE__, __LINE__);                               \
  }

inline void sched_gpu_assert(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDACHK: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

INLINE int smid()
{
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return (int)smid;
}

INLINE int nsmid()
{
  uint32_t smid;
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(smid));
  return (int)smid;
}
#endif

// https://stackoverflow.com/questions/38088732/explanation-to-aligned-malloc-implementation
INLINE void* aligned_malloc(size_t required_bytes, size_t alignment)
{
  void* p1;   // original block
  void** p2;  // aligned block
  int offset = alignment - 1 + sizeof(void*);
  if ((p1 = (void*)malloc(required_bytes + offset)) == NULL) {
    return NULL;
  }
  p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
  p2[-1] = p1;
  return p2;
}
void aligned_free(void* p) { free(((void**)p)[-1]); }
