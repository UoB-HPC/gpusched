#pragma once

#include <cstdint>

#include <atomics.h>

struct alignas(CACHELINE_SIZE) lock_t {
  volatile uint32_t lck_val;
  volatile uint32_t rel_val;
};

INLINE void init_lock(lock_t* l)
{
  l->lck_val = 0;
  l->rel_val = 0;
}

INLINE void lock(lock_t* l)
{
  uint32_t ticket = atomic_add(&l->lck_val, 1u);
  // memfence
  while (1) {
    uint32_t currently_serving = atomic_add(&l->rel_val, 0u);
    // memfence
    if (ticket == currently_serving) break;
    // cpu_yield
  }
}
INLINE void unlock(lock_t* l)
{
  __threadfence();
  atomic_add(&l->rel_val, 1u);
}
