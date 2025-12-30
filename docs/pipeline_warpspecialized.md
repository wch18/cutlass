# A Beginner's Tour of CUTLASS 3.x GEMM Pipelines (Hopper+/SM90)

New to CUTLASS and wondering how the GEMM mainloop keeps Tensor Cores busy while data streams from global memory? This guide walks through the pipeline helpers in `cutlass/pipeline`, shows where warp-specialization fits, and uses annotated code snippets and diagrams so you can connect the high-level concepts to real headers.

> **Scope.** The examples reference the SM90+ pipeline utilities introduced for Hopper-class GPUs. Older architectures use different helpers but the ideas (producer/consumer stages and barrier hand-offs) are similar.

---

## 1) Why a pipeline exists at all

A GEMM kernel juggles two expensive activities:

1. **Loading tiles of A/B/C** from global memory into shared memory (SMEM) using TMA or `cp.async`.
2. **Issuing Tensor Core MMA** on already-loaded tiles.

Rather than serialize those tasks, CUTLASS sets up a **producer/consumer pipeline** so memory transfers for the next tile overlap with math on the current tile. Depth is controlled by the `Stages` template parameter in the mainloop.

### Minimal mental model
```
while (k_tiles_left) {
  producer loads tile[k + pipeline_depth];
  consumer computes MMA on tile[k];
  barriers ensure compute never reads a slot while producer overwrites it;
}
```

---

## 2) The core building blocks

### 2.1 `PipelineState`
Tracks the circular stage index (`stage_idx`), a phase bit for barriers (`phase`), and a running iteration count. Advancing rolls over the stage and toggles phase so producer/consumer stay in sync across the ring buffer:

```cpp
CUTLASS_DEVICE
PipelineState& advance(uint32_t num_iterations) {
  if constexpr (Stages > 0) {
    // Number of iterations cross over the stage boundary => flipped phase
    if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages ) {
      phase_ ^= 1;
    }
    // How many times number of iterations cross over the stage boundary and
    // end up on a odd number => flipped phase
    if ((num_iterations >= Stages) && (((index_ + num_iterations) / Stages) % 2) == 1) {
      phase_ ^= 1;
    }
    index_ = (index_ + num_iterations) % Stages;
    count_ += num_iterations;
  }
  return *this;
}

template<class Pipeline>
CUTLASS_DEVICE
PipelineState<Pipeline::Stages> make_producer_start_state() {
  // Producer starts with an opposite phase as the buffers are initially empty
  constexpr int InitialProducerStage = 0;
  constexpr uint32_t InitialProducerPhase = 1;
  constexpr uint32_t InitialProducerCount = 0;
  return {InitialProducerStage, InitialProducerPhase, InitialProducerCount};
}
```
[Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

### 2.2 Barrier-backed TMA pipeline (`PipelineTmaAsync`)
Used by SM90 GEMM/conv mainloops that move tiles with TMA:

* **Barriers:** Paired arrays `full_barrier_` (consumer waits) and `empty_barrier_` (producer waits) live in SMEM. Initialization seeds arrival counts for every stage and, on clusters, distributes which threads signal each destination block to reduce contention:

  ```cpp
  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  template <class ClusterShape>
  CUTLASS_DEVICE
  static void init_barriers(SharedStorage& storage, Params params, ClusterShape cluster_shape) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == 0);
    is_initializing_warp = (warp_idx == params.initializing_warp);
    if (is_initializing_warp) {
      uint32_t const producer_arv_cnt = params.num_producers;
      uint32_t const num_consumer_warpgroups_per_cluster = cute::ceil_div(params.num_consumers, static_cast<uint32_t>(NumThreadsPerWarpGroup));
      uint32_t multicast_consumer_arrival_count = params.num_consumers; // If cluster_size is 1
      if (cute::size(cluster_shape) > 1) {
        multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
              num_consumer_warpgroups_per_cluster;
      }
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt, multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }
  ```
  [Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

* **Producer path:** `producer_try_acquire`/`producer_acquire` wait on `empty_barrier_` for a free slot; the leader thread issues `tma::expect` and `tma::commit` (or `producer_commit`), then advances the pipeline state:

  ```cpp
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token != BarrierStatus::WaitDone) {
      empty_barrier_ptr_[stage].wait(phase);
    }
    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
  }
  ```
  [Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

* **Consumer path:** `consumer_wait` waits on `full_barrier_`, does MMA on the tile for that stage, then `consumer_release` arrives on `empty_barrier_` so the producer can reuse it:

  ```cpp
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
  }
  ```
  [Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

### 2.3 Store pipeline (`PipelineTmaStore`)
Mirrors the TMA load pipeline for epilogues, letting you throttle how many store batches are in flight (`UnacquiredStages`) before blocking, which overlaps writeback with math:

```cpp
template <int Stages_, int UnacquiredStages_ = Stages_-1>
class PipelineTmaStore {
public:
  static constexpr uint32_t Stages = Stages_;
  static constexpr uint32_t UnacquiredStages = static_cast<uint32_t>(UnacquiredStages_);
  using PipelineState = cutlass::PipelineState<Stages>;

  struct Params { bool always_wait = false; };

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state) { producer_acquire(state.index(), state.count()); }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) { producer_commit(state.index(), state.count()); }

  CUTLASS_DEVICE
  void producer_tail([[maybe_unused]] PipelineState state) { tma_store_wait<0>(); }

private:
  Params params_;
  CUTLASS_DEVICE
  void producer_acquire([[maybe_unused]] uint32_t stage, uint32_t count) {
    if (params_.always_wait || count > UnacquiredStages) { tma_store_wait<UnacquiredStages>(); }
  }

  CUTLASS_DEVICE
  void producer_commit([[maybe_unused]] uint32_t stage, [[maybe_unused]] uint32_t count) {
    tma_store_arrive();
  }
};
```
[Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

### 2.4 Lightweight async pipeline (`PipelineAsync`)
For kernels using `cp.async` or non-TMA traffic, `PipelineAsync` provides the same acquire/commit/wait/release choreography on plain cluster barriers, parameterized by producer/consumer roles. The class mirrors the TMA helpers but with a simpler initialization:

```cpp
template <int Stages_>
class PipelineAsync {
public:
  struct SharedStorage {
    FullBarrier full_barrier_[Stages];
    EmptyBarrier empty_barrier_[Stages];
  };

  static CUTLASS_DEVICE void init_barriers(SharedStorage& storage, Params params) {
    int warp_idx = canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == params.initializing_warp);
    if (is_initializing_warp) {
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, params.producer_arv_count, params.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();
  }
```
[Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

It exposes the same wrapper helpers as the TMA version (forwarding to the private implementations below) and adds a `producer_tail` guard so producer blocks in a cluster don’t exit early while others still depend on them:

```cpp
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) {
    producer_commit(state.index());
  }

  template<class UserDefinedArriveOp>
  CUTLASS_DEVICE
  void producer_commit(PipelineState state, UserDefinedArriveOp&& user_defined_arrive_op) {
    cute::forward<UserDefinedArriveOp>(user_defined_arrive_op)(producer_get_barrier(state.index()));
    producer_commit(state);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token = {BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) {
    consumer_release(state.index());
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }
```
[Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

The private implementations mirror the TMA pipeline but without TMA-specific arrive/expect calls—producers and consumers simply wait on the appropriate barrier, then arrive to flip it for the opposite side:

```cpp
  CUTLASS_DEVICE
  ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token) {
    detail::pipeline_check_is_producer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].arrive();
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    detail::pipeline_check_is_consumer(params_.role);
    bool done = full_barrier_ptr_[stage].test_wait(phase);
    if (!done) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token) {
    detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid);
  }
};
```
[Source: `include/cutlass/pipeline/sm90_pipeline.hpp`](../include/cutlass/pipeline/sm90_pipeline.hpp)

---

## 3) Seeing the pipeline inside a GEMM mainloop

A typical SM90 GEMM mainloop specialization wires the pipeline into a loop over K tiles. The pseudo-flow matches the real CUTLASS helpers (comments reference the header so you can explore further):

```cpp
// Pseudocode distilled from SM90 GEMM mainloops
PipelineState ps = PipelineState::make_initial(kStages); // stage=0, phase=0
pipeline.producer_arrive(ps); // pre-fill barriers

for (int k_iter = 0; k_iter < kTileCount; ++k_iter) {
  // 1) Producer warp: acquire a free stage and launch TMA copy
  if (is_producer) {
    pipeline.producer_acquire(ps);                    // waits on empty barrier
    issue_tma_load(ps.stage_idx);                     // tma::expect/commit
    pipeline.producer_commit(ps);                     // marks full barrier
  }

  // 2) Consumer warps: wait for data, then MMA
  if (is_consumer) {
    pipeline.consumer_wait(ps);                       // waits on full barrier
    mma_on_tile(ps.stage_idx);                         // Tensor Core MMA
    pipeline.consumer_release(ps);                     // signals empty barrier
  }

  // 3) Everyone advances to the next circular stage
  ps = pipeline.advance(ps);
}
```

All real SM90 mainloops in CUTLASS follow this shape—the differences are **who plays producer**, how many stages exist, and whether TMA or `cp.async` drives movement.

---

## 4) How warp specialization plugs in

The dispatch policy chooses both **pipeline flavor** and **warp roles**. For GEMM, you will see tags in `cutlass/gemm/dispatch_policy.hpp`:

* `KernelTmaWarpSpecialized` → one producer warp, multiple consumer warps.【F:include/cutlass/gemm/dispatch_policy.hpp†L70-L84】
* `KernelTmaWarpSpecializedPingpong` → two producer warps that alternate by stage (ping-pong).【F:include/cutlass/gemm/dispatch_policy.hpp†L84-L96】
* `KernelTmaWarpSpecializedCooperative` → producer duties are distributed across a cluster, letting several CTAs feed the same shared tiles.【F:include/cutlass/gemm/dispatch_policy.hpp†L96-L122】

The convolution dispatcher (`cutlass/conv/dispatch_policy.hpp`) forwards similar tags for implicit GEMM conv kernels, so the same pipeline machinery is reused.【F:include/cutlass/conv/dispatch_policy.hpp†L47-L74】

### Visualizing warp-specialized flow (two-stage example)

```mermaid
graph LR
  subgraph Stage0
    P0[Producer warp
    TMA tile k] -->|full barrier| C0[Consumer warps
    MMA tile k-1]
  end
  subgraph Stage1
    P1[Producer warp
    TMA tile k+1] -->|full barrier| C1[Consumer warps
    MMA tile k]
  end
  C0 -->|empty barrier| P0
  C1 -->|empty barrier| P1
  P0 -.advance.-> P1
  P1 -.advance.-> P0
```

Producer warps stay in the TMA-heavy code path; consumer warps stay in the MMA-heavy path, reducing divergence and keeping Tensor Cores saturated once the pipe is full.

---

## 5) Understanding `Stages` and common patterns

`Stages` controls how many shared-memory buffers (and barrier pairs) the pipeline rotates through. CUTLASS defaults vary by architecture and problem shape, but the trade-offs are consistent:

* **Two-stage (double buffering)** – simplest ring. One stage is feeding MMA while the other is being filled. Hides most global→SMEM latency when TMA and MMA times are similar.
* **Three+ stages (deep buffer)** – allows multiple outstanding TMA copies. Helps when global-memory latency dominates (large K, high operand reuse) so the consumer rarely stalls waiting for data.
* **Ping-pong producer schedules** – when a single producer warp would be underutilized, alternating producer roles every stage keeps more lanes busy issuing TMA traffic at the cost of extra role bookkeeping.
* **Cluster-cooperative schedules** – for kernels that multicast tiles across CTAs, cooperative producers reduce redundant global loads and mask inter-CTA multicast latency. Barrier initialization accounts for which thread signals each destination block to avoid contention.【F:include/cutlass/pipeline/sm90_pipeline.hpp†L305-L376】
* **Epilogue store pipeline depth** – increasing `UnacquiredStages` in `PipelineTmaStore` overlaps global stores with remaining MMA/epilogue math, hiding writeback latency when C is large.【F:include/cutlass/pipeline/sm90_pipeline.hpp†L646-L708】

---

## 6) What latency is being hidden?

* **Global → shared transfer time** – `producer_acquire` waits for an empty slot; once TMA is in flight, the consumer keeps using older stages until the full barrier flips, masking the transfer.
* **Tensor Core issue latency** – deeper `Stages` keep a backlog of ready tiles so the consumer warp never bubbles waiting for data.
* **Shared-memory bank conflicts / barrier overhead** – distributing signaling threads and per-stage barriers reduces hot spots when multiple CTAs participate, masking some synchronization delay.【F:include/cutlass/pipeline/sm90_pipeline.hpp†L343-L376】
* **Global store completion** – store pipelines allow several write batches to proceed before blocking, overlapping writeback with ongoing math.【F:include/cutlass/pipeline/sm90_pipeline.hpp†L646-L708】

---

## 7) How to explore the code yourself

1. Start in `include/cutlass/pipeline/sm90_pipeline.hpp` for the barrier primitives and pipeline helper classes.
2. Browse GEMM dispatch tags in `include/cutlass/gemm/dispatch_policy.hpp` and map them to the pipeline flavor.
3. Look at an SM90 mainloop instantiation (e.g., `cutlass/gemm/kernel/sm90_gemm.hpp`) and search for `Stages` or `PipelineTmaAsync` to see how a concrete kernel wires producers and consumers.

Armed with these references, you can tune `Stages`, choose a dispatch policy, and understand exactly how CUTLASS keeps Tensor Cores fed on Hopper-class GPUs.
