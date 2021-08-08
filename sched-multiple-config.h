#pragma once

#if !defined(SCHED_WFFF) && !defined(SCHED_WFFL) && !defined(SCHED_WFLF) &&    \
    !defined(SCHED_WFLL)
#define SCHED_WFLF
#endif

#ifdef SCHED_WFFF
INLINE task_t* dequeue_this(worker_t* worker) { return dequeue_front(worker); }
INLINE task_t* dequeue_other(worker_t* worker) { return dequeue_front(worker); }
#endif

#ifdef SCHED_WFFL
INLINE task_t* dequeue_this(worker_t* worker) { return dequeue_front(worker); }
INLINE task_t* dequeue_other(worker_t* worker) { return dequeue_back(worker); }
#endif

#ifdef SCHED_WFLF
INLINE task_t* dequeue_this(worker_t* worker) { return dequeue_back(worker); }
INLINE task_t* dequeue_other(worker_t* worker) { return dequeue_front(worker); }
#endif

#ifdef SCHED_WFLL
INLINE task_t* dequeue_this(worker_t* worker) { return dequeue_back(worker); }
INLINE task_t* dequeue_other(worker_t* worker) { return dequeue_back(worker); }
#endif

//#if defined (SHCED_WFFL) || defined (SCHED_WFLL)
// INLINE task_t* dequeue_this(worker_t* worker) { return dequeue_back(worker);
// } #else INLINE task_t* dequeue_this(worker_t* worker) { return
// dequeue_front(worker); } #endif #if defined (SCHED_WFFF) || defined
//(SCHED_WFFL) INLINE task_t* dequeue_other(worker_t* worker) { return
// dequeue_front(worker); } #else INLINE task_t* dequeue_other(worker_t* worker)
// { return dequeue_back(worker); } #endif

#if !defined(STEAL_RANDOM) &&                                                  \
    !defined(STEAL_HIERARCHICAL) && !defined(STEAL_ITERATIVE) &&               \
    !defined(STEAL_HIERARCHICAL_ITERATIVE) &&                                  \
    !defined(STEAL_HIERARCHICAL_RANDOM)
#define STEAL_RANDOM
#endif

#ifdef STEAL_RANDOM
INLINE int32_t find_victim(worker_t* worker, int32_t worker_id,
                           int32_t num_workers)
{
  int32_t victim_id = get_random(worker) % (num_workers - 1);
  if (victim_id >= worker_id) victim_id++;
  return victim_id;
}
//INLINE task_t* steal(worker_t* worker, int32_t worker_id, int32_t num_workers)
//{
//  int32_t victim_id = get_random(worker) % (num_workers - 1);
//  if (victim_id >= worker_id) victim_id++;
//  worker_t* victim = worker->team->workers + victim_id;
//  return dequeue_other(victim);
//}
#endif

#ifdef STEAL_ITERATIVE
INLINE int32_t find_victim(worker_t* worker, int32_t worker_id,
                           int32_t num_workers)
{
  for (int i = (worker_id + 1) % num_workers; i != worker_id;
       i = (i + 1) % num_workers) {
    worker_t* victim = worker->team->workers + i;
    if (victim->queue->ntasks > 0) return i;
  }

  // if we didn't find any queue with tasks, just return a random victim
  int32_t victim_id = get_random(worker) % (num_workers - 1);
  if (victim_id >= worker_id) victim_id++;
  return victim_id;
}
#endif

#ifdef STEAL_HIERARCHICAL_RANDOM
INLINE int32_t find_victim(worker_t* worker, int32_t worker_id, int32_t num_workers)
{
  const uint32_t sm_id = smid();
  const uint32_t nsm_id = nsmid();
  const int num_blocks_per_sm = worker->team->num_blocks / nsm_id;
  const int num_blocks = worker->team->num_blocks;

  for (int i = 1; i < num_blocks_per_sm; ++i) {
    int32_t vid = (i * nsm_id + worker_id) % num_blocks;
    worker_t* victim = worker->team->workers + vid;
    if (victim->queue->ntasks > 0) return vid;
  }

  int32_t victim_id = get_random(worker) % (num_workers - 1);
  if (victim_id >= worker_id) victim_id++;
  return victim_id;
}
//INLINE task_t* steal(worker_t* worker, int32_t* victim_id, int32_t num_workers)
//{
//  const uint32_t worker_id = worker->id;
//  const uint32_t sm_id = smid();
//  const uint32_t nsm_id = nsmid();
//  const int num_blocks_per_sm = worker->team->num_blocks / nsm_id;
//  const int num_blocks = worker->team->num_blocks;
//
//  task_t* t;
//
//  // iterate over all blocks on same SM
//  for (int i = 1; i < num_blocks_per_sm; ++i) {
//    int32_t vid = (i * nsm_id + worker_id) % num_blocks;
//    worker_t* victim = worker->team->workers + vid;
//    t = dequeue_other(victim);
//    if (t != nullptr) return t;
//  }
//
//  *victim_id = get_random(worker) % (num_workers - 1);
//  if (*victim_id >= worker_id) (*victim_id)++;
//  worker_t* victim = worker->team->workers + *victim_id;
//  return dequeue_other(victim);
//}
#endif

#ifdef STEAL_HIERARCHICAL_ITERATIVE
INLINE int32_t find_victim(worker_t* worker, int32_t worker_id, int32_t num_workers)
{
  const uint32_t sm_id = smid();
  const uint32_t nsm_id = nsmid();
  const int num_blocks_per_sm = worker->team->num_blocks / nsm_id;
  const int num_blocks = worker->team->num_blocks;

  for (int i = 1; i < num_blocks_per_sm; ++i) {
    int32_t vid = (i * nsm_id + worker_id) % num_blocks;
    worker_t* victim = worker->team->workers + vid;
    if (victim->queue->ntasks > 0) return vid;
  }

  for (int i = (worker_id + 1) % num_workers; i != worker_id;
       i = (i + 1) % num_workers) {
    worker_t* victim = worker->team->workers + i;
    if (victim->queue->ntasks > 0) return i;
  }

  int32_t victim_id = get_random(worker) % (num_workers - 1);
  if (victim_id >= worker_id) victim_id++;
  return victim_id;
}
//INLINE task_t* steal(worker_t* worker, int32_t* victim_id, int32_t num_workers)
//{
//  const uint32_t worker_id = worker->id;
//  const uint32_t sm_id = smid();
//  const uint32_t nsm_id = nsmid();
//  const int num_blocks_per_sm = worker->team->num_blocks / nsm_id;
//  const int num_blocks = worker->team->num_blocks;
//
//  task_t* t;
//
//  // iterate over all blocks on same SM
//  for (int i = 1; i < num_blocks_per_sm; ++i) {
//    int32_t vid = (i * nsm_id + worker_id) % num_blocks;
//    worker_t* victim = worker->team->workers + vid;
//    t = dequeue_other(victim);
//    if (t != nullptr) return t;
//  }
//  for (int i = (worker_id + 1) % num_workers; i != worker_id;
//       i = (i + 1) % num_workers) {
//    worker_t* victim = worker->team->workers + i;
//    t = dequeue_other(victim);
//    if (t != nullptr) return t;
//  }
//  return nullptr;
//}
#endif
