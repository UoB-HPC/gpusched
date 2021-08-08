#pragma once

#include <alloc.h>
#include <lb-queue.h>
#include <task.h>
#include <utils.h>
#include <worker.h>

#include <sched-multiple-config.h>

#ifdef SCHED_MEMORISE
INLINE void execute_tasks(worker_t* worker, volatile uint32_t* exit_cond,
                          const uint32_t exit_val)
{
  int32_t worker_id = gpu_utils::global_worker_id();
  int32_t victim_id = -2;
  int32_t num_workers = NWORKERS * worker->team->num_blocks;
  int32_t use_own_tasks = 1;
  int32_t new_victim = 0;
  task_t* t;

  while (1) {
    int nt;
    if (gpu_utils::thread_id() == 0) nt = atomic_add(exit_cond, 0u);
    nt = gpu_utils::broadcast(nt, 0);
    if (nt <= exit_val) break;

    t = NULL;
    if (gpu_utils::thread_id() == 0) {
      if (use_own_tasks) {
        t = dequeue_this(worker);
      }
      if (t == NULL) {
        use_own_tasks = 0;
        if (victim_id == -2) {
          victim_id = worker->last_stolen;
        }
        if (victim_id == -1 && !new_victim) {
          victim_id = find_victim(worker, worker_id, num_workers);
          // printf("(%d) victim id = %d\n", worker_id, victim_id);
        }
        // steal
        // printf("(%d) victim id = %d new victim = %d\n", worker_id, victim_id,
        // new_victim);
        worker_t* victim = worker->team->workers + victim_id;
        t = dequeue_other(victim);

        if (t != NULL) {
          if (worker->last_stolen != victim_id) {
            worker->last_stolen = victim_id;
            new_victim = 1;
          }
        }
        else {
          worker->last_stolen = -1;
          victim_id = -2;
        }
      }
    }

    t = (task_t*)gpu_utils::broadcast((unsigned long long)t, 0ull);

    if (t != nullptr) {
      execute_task(worker, t);
      finish_task(worker, t);

      if (gpu_utils::thread_id() == 0 && !use_own_tasks &&
          worker->queue->ntasks != 0) {
        use_own_tasks = 1;
        new_victim = 0;
      }
    }
    else {  // effectively we've failed to pop or steal a task, reset
            // memorisation (eqv. to exiting LLVM OpenMP task loop)
      victim_id = -2;
      use_own_tasks = 1;
      new_victim = 0;
    }
  }
}
#else
INLINE void execute_tasks(worker_t* worker, volatile uint32_t* exit_cond,
                          const uint32_t exit_val)
{
  int32_t worker_id = gpu_utils::global_worker_id();
  int32_t victim_id = -2;
  int32_t num_workers = NWORKERS * worker->team->num_blocks;

  while (1) {
    int nt;
    if (gpu_utils::thread_id() == 0) nt = atomic_add(exit_cond, 0u);
    nt = gpu_utils::broadcast(nt, 0);
    if (nt <= exit_val) break;

    task_t* t;
    if (gpu_utils::thread_id() == 0) {
      // try our own queue
      t = dequeue_this(worker);
      // randomly try another queue
      if (t == nullptr) {
        victim_id = find_victim(worker, worker_id, num_workers);
        worker_t* victim = worker->team->workers + victim_id;
        t = dequeue_other(victim);
      }
    }

    t = (task_t*)gpu_utils::broadcast((unsigned long long)t, 0ull);
    //__threadfence();

    if (t != nullptr) {
      execute_task(worker, t);
      finish_task(worker, t);
    }
  }
  // if (gpu_utils::thread_id() == 0)
  //  printf("num own q = %d, num stolen = %d\n", num_own_q, num_stolen);
}
#endif

INLINE void generate_task(worker_t* worker, task_func_t func,
                          uint32_t nprivates, void** privates)
{
  task_t* task;
  bool success;
  if (gpu_utils::thread_id() == 0) {
    task = alloc_task(worker, func, nprivates, privates);
    success = enqueue_back(worker, task);
  }

  success = gpu_utils::broadcast(success, 0);
  if (!success) {
    task = (task_t*)gpu_utils::broadcast((unsigned long long)task, 0);
    // we could not enqueue the task, so just run it immediately
    execute_task(worker, task);
  }
  else {
    if (gpu_utils::thread_id() == 0) {
      // as we enqueued a task we must record it
      atomic_add(&worker->team->num_barrier_tasks, 1u);
      atomic_add(&task->parent->num_children, 1u);
    }
  }
}

INLINE void taskwait(worker_t* worker)
{
  task_t* current_task = worker->current_task;
  volatile uint32_t* num_children = &worker->current_task->num_children;
  // printf("taskwait %d has %d children\n", worker->id, *num_children);
  execute_tasks(worker, num_children, 0u);
  worker->current_task = current_task;
}

INLINE void barrier(worker_t* worker, int num_workers = -1)
{
  num_workers =
      (num_workers == -1) ? worker->team->num_blocks * NWORKERS : num_workers;
  volatile uint32_t* barrier_counter = &worker->team->barrier_counter;
  const uint32_t barrier_counter_old = worker->team->barrier_counter - 1;

  taskwait(worker);

  int last = 0;
  if (num_workers <= 1) return;

  if (gpu_utils::thread_id() == 0) {
    const uint32_t bv = atomic_add(&worker->team->barrier_val, 1u);
    if (bv >= num_workers - 1) {
      worker->team->barrier_val = 0;
      __threadfence();
      worker->team->barrier_counter--;
      last = 1;
    }
  }
  last = gpu_utils::broadcast(last, 0);

  if (!last) {
    task_t* current_task = worker->current_task;
    execute_tasks(worker, barrier_counter, barrier_counter_old);
    worker->current_task = current_task;
  }
}

template <task_func_t func>
__global__ void set_initial_tasks(team_t* team, uint32_t nprivates,
                                  void** privates)
{
  int worker_id = blockIdx.x;
  worker_t* worker = team->workers + worker_id;
  task_t* task = alloc_task(worker, func, nprivates, privates);
  if (!task) {
    printf("error, could not allocate initial_task");
    return;
  }

  if (!enqueue_back(worker, task)) {
    printf("error, could not enqueue initial task");
    return;
  }

  atomic_add(&team->num_barrier_tasks, 1u);
}

__global__ void fork_team_device(team_t* team)
{
  worker_t* worker = team->workers + gpu_utils::global_worker_id();
  // run until we exhaust all the teams tasks
  execute_tasks(worker, &team->num_barrier_tasks, 0u);
}

template <task_func_t func>
void fork_team(team_t* h_team, team_t* d_team, uint32_t nprivates,
               void** privates, int num_initial_tasks = 1,
               size_t shmem_size_per_worker = 0)
{
  cudaDeviceProp prop;
  CUDACHK(cudaGetDeviceProperties(&prop, 0));
  const int num_sms = prop.multiProcessorCount;

  int max_blocks = 1;
  CUDACHK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, fork_team_device, NWORKERS * gpu_utils::warp_size,
      shmem_size_per_worker));

  if (h_team->num_blocks > max_blocks * num_sms) {
    fprintf(stderr,
            "error: num. blocks (%d) exceeds max active blocks (%d) for this "
            "task.\n",
            h_team->num_blocks, max_blocks * num_sms);
    exit(1);
  }

  void** d_privates;
  CUDACHK(cudaMalloc(&d_privates, sizeof(void*) * nprivates));
  CUDACHK(cudaMemcpy(d_privates, privates, sizeof(void*) * nprivates,
                     cudaMemcpyHostToDevice));

  set_initial_tasks<func>
      <<<num_initial_tasks, 1>>>(d_team, nprivates, d_privates);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());

  fork_team_device<<<h_team->num_blocks, NWORKERS * NTHREADS,
                     shmem_size_per_worker * NWORKERS>>>(d_team);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());
}

__global__ void create_team_device(team_t* team, void* malloc_storage,
                                   uint64_t num_cachelines_per_sm)
{
  if (gpu_utils::thread_id() == 0) {
    worker_t* worker = &team->workers[gpu_utils::global_worker_id()];
    worker->id = gpu_utils::global_worker_id();
    worker->team = team;
    worker->current_task = nullptr;
    // Does not happen often, okay to do (slow) device-side malloc
    worker->queue = (queue_t*)aligned_malloc(sizeof(queue_t), 128);
    init_queue(worker->queue);
    init_random(worker);

#ifdef SCHED_MEMORISE
    worker->last_stolen = -1;
#endif

    if (gpu_utils::global_worker_id() == 0) {
      team->num_barrier_tasks = 0;
      team->barrier_val = 0;
      team->barrier_counter = 0xffffffff;
    }
  }
}

void create_team(int nblocks, uint64_t num_cachelines_per_sm, team_t** h_team,
                 team_t** d_team, uint64_t num_cachelines_master = 0)
{
  // if master cachelines not set, use same as other SMs
  if (num_cachelines_master == 0) num_cachelines_master = num_cachelines_per_sm;

  cudaDeviceProp prop;
  CUDACHK(cudaGetDeviceProperties(&prop, 0));
  const int num_sms = prop.multiProcessorCount;

  const size_t stack_size = 32768;
  const size_t heap_size = (QUEUE_SIZE + 4 * CACHELINE_SIZE + sizeof(queue_t)) *
                           NWORKERS * nblocks * sizeof(void*);

  printf("heap size = %zu\n", heap_size);

  CUDACHK(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  CUDACHK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));

  *h_team = (team_t*)malloc(sizeof(team_t));
  (*h_team)->num_blocks = nblocks;
  //(*h_team)->max_alloc_size = num_cachelines_per_sm * CACHELINE_SIZE;

  CUDACHK(
      cudaMalloc(&((*h_team)->workers), sizeof(worker_t) * NWORKERS * nblocks));
  CUDACHK(
      cudaMalloc(&((*h_team)->malloc_info), sizeof(malloc_info_t) * num_sms));

  void* malloc_storage;
  CUDACHK(
      cudaMalloc(&malloc_storage,
                 CACHELINE_SIZE * num_cachelines_master +
                     CACHELINE_SIZE * num_cachelines_per_sm * (num_sms - 1)));

  printf("Master has %d CL, Other has %d\n", num_cachelines_master,
         num_cachelines_per_sm);

  malloc_info_t* h_malloc_info =
      (malloc_info_t*)malloc(sizeof(malloc_info_t) * num_sms);

  h_malloc_info[0].head = h_malloc_info[0].tail = (char*)malloc_storage;
  h_malloc_info[0].max_alloc_size = num_cachelines_master * CACHELINE_SIZE;
  // set the offset for SMs other than master
  char* master_offset =
      (char*)malloc_storage + num_cachelines_master * CACHELINE_SIZE;

  for (int i = 1; i < num_sms; ++i) {
    h_malloc_info[i].head = h_malloc_info[i].tail =
        master_offset + CACHELINE_SIZE * num_cachelines_per_sm * (i - 1);
    h_malloc_info[i].max_alloc_size = num_cachelines_per_sm * CACHELINE_SIZE;
  }

  CUDACHK(cudaMemcpy((*h_team)->malloc_info, h_malloc_info,
                     sizeof(malloc_info_t) * num_sms, cudaMemcpyHostToDevice));

  CUDACHK(cudaMalloc(d_team, sizeof(team_t)));
  CUDACHK(cudaMemcpy(*d_team, *h_team, sizeof(team_t), cudaMemcpyHostToDevice));

  printf("num sms = %d num blocks = %d, nworkerstot = %d\n", num_sms, nblocks,
         NWORKERS * nblocks);

  create_team_device<<<nblocks, NTHREADS * NWORKERS>>>(*d_team, malloc_storage,
                                                       num_cachelines_per_sm);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());

  free(h_malloc_info);
}

__global__ void reset_team_device(team_t* team, int num_sms)
{
  const int worker_id = blockIdx.x;
  if (worker_id < num_sms) {
    team->malloc_info[worker_id].tail = team->malloc_info[worker_id].head;
  }

  if (worker_id == 0) {
    team->num_barrier_tasks = 0;
    team->barrier_val = 0;
    team->barrier_counter = 0xffffffff;
  }

  worker_t* worker = team->workers + worker_id;
  worker->current_task = nullptr;
  reset_queue(worker->queue);
  // is this really needed? will just make workers steal from same targets in
  // same order in subsequent parallel regions
  init_random(worker);

#ifdef SCHED_MEMORISE
  worker->last_stolen = -1;
#endif
}

void reset_team(team_t* h_team, team_t* d_team)
{
  cudaDeviceProp prop;
  CUDACHK(cudaGetDeviceProperties(&prop, 0));
  const int num_sms = prop.multiProcessorCount;
  const int num_workers = NWORKERS * h_team->num_blocks;

  reset_team_device<<<num_workers, 1>>>(d_team, num_sms);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());
}
