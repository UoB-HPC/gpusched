#pragma once

#include <utils.h>

INLINE void mem_fence()
{
#ifdef __CUDA_ARCH__
  __threadfence();
#else
  __sync_synchronize();
#endif
}

template <class T>
INLINE T atomic_compare_exchange(volatile T* addr, T compare, T val)
{
#ifdef __CUDA_ARCH__
  return atomicCAS((T*)addr, compare, val);
#else
  return (T)__sync_val_compare_and_swap(addr, compare, val);
#endif
}

template <class T>
INLINE T atomic_exchange(volatile T* addr, T val)
{
#ifdef __CUDA_ARCH__
  return atomicExch((T*)addr, val);
#else
  T ret;
  __atomic_exchange(addr, &val, &ret, __ATOMIC_ACQ_REL);
  return ret;
#endif
}

template <class T>
INLINE T atomic_add(volatile T* addr, const T val)
{
#ifdef __CUDA_ARCH__
  return atomicAdd((T*)addr, val);
#else
  return __sync_fetch_and_add(addr, val);
#endif
}

template <class T>
INLINE T atomic_sub(volatile T* addr, const T val)
{
#ifdef __CUDA_ARCH__
  return atomicSub((T*)addr, val);
#else
  return __sync_fetch_and_sub(addr, val);
#endif
}
