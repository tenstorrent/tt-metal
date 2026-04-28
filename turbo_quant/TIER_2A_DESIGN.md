# Tier 2A — cross-core chunk distribution (design doc)

Goal: split the per-(batch, q_head) chunk loop across `num_cores_per_head`
worker cores; merge their partial running (max, sum, output) via cross-core
reduce. Mirror the pattern in
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`.

**Expected gain:** at 32K, per-call drops from ~70 ms (1 core/head, sequential)
to ~9 ms (8 cores/head, parallel) — 8× speedup. Compounds with the other
optimizations.

**Memory:** unchanged (0.75× of BFP8 baseline).

---

## Why this works for TQ FD

The current kernel runs one (B, NQH) tuple per core. With `B=1, NQH=32` (Llama
8B per device), the program uses 32 of 56 cores; 24 sit idle. On T3K with 1 KV
head per device and 4 Q heads/device, only 4 cores work — 52 idle. Splitting
each tuple's chunk loop across `K = floor(num_idle_cores / num_active_cores)`
extra workers extracts that idle parallelism.

For Llama 8B:
- N150 (1 device, 32 NQH): 56/32 = 1 → no parallelism gain on N150 single
  device.
- T3K (8 devices, 4 NQH/device): 56/4 = 14 → up to 14× per device.
- Galaxy (32 devices, 1 NQH/device): 56/1 = 56 → up to 56× per device.

So 2A is primarily a multi-device long-context win. N150 needs Tier 1A
(fused dequant-matmul) to make per-call faster.

---

## Algorithm

For each (batch, q_head) tuple, with `K` worker cores per tuple:

1. **Worker `w` (w in [0, K))** processes chunk range
   `[w * chunks_per_worker, (w+1) * chunks_per_worker)` — its own slice of
   the cur_pos chunks. It runs the existing online-softmax loop on its
   slice. Output: per-worker `(max_w, sum_w, out_w)` in L1.
2. **All workers** signal a per-tuple semaphore on completion.
3. **Reducer (worker 0)** waits on the semaphore (count == K), then walks
   workers 1..K-1, pulling each one's `(max_w, sum_w, out_w)` from the
   neighbor's L1 via NoC, and merges via the standard online-softmax
   correction:
   ```
   new_max = max(prev_max, w_max)
   alpha = exp((prev_max - new_max) * scale)
   beta  = exp((w_max - new_max) * scale)
   new_sum = alpha * prev_sum + beta * w_sum
   new_out = alpha * prev_out + beta * w_out
   ```
4. **Reducer** finalises: `new_out / new_sum`, writes via the existing
   writer.

This is the "linear all-to-one" reduce. The standard sdpa_decode does
*tree* reduction in `ceil(log2(K))` rounds — strictly faster but more
complex. Linear is good enough for K ≤ 16 (T3K case) where reduce cost is
already <10% of chunk-loop cost.

---

## Plumbing required

### A. Device op attribute
`SDPATQDeviceOperation::operation_attributes_t` gets a new field:
```cpp
uint32_t num_cores_per_head = 1;  // 1 = current behavior
```
Plumbed through Python binding (default 1). User-facing: optional kwarg on
`ttnn.experimental.turbo_quant_sdpa_decode`. Internal: program factory
clamps to `min(num_idle_cores / num_active_work_items, requested)`.

### B. Program factory
- New work-distribution math:
  ```cpp
  uint32_t num_active_work_items = B * NQH;       // current
  uint32_t cores_per_head = std::min(
      attrs.num_cores_per_head,
      num_cores / num_active_work_items);
  uint32_t total_active_cores = num_active_work_items * cores_per_head;
  ```
- Each (B, NQH) tuple gets `cores_per_head` consecutive cores.
- Compute & write per-core runtime args:
  - `core_idx_in_group`: 0..cores_per_head-1
  - `cores_per_head_for_this_group`: same for all cores in a group
  - `is_reducer`: 1 iff core_idx_in_group == 0
  - `worker_neighbor_x[K-1]`, `worker_neighbor_y[K-1]`: NoC coords of the
    other workers (for the reducer to pull partial state)

### C. Reader changes
`reader_tq_decode.cpp` already reads chunks `[0, valid_k_chunks)` based on
cur_pos. Add a per-core slice:
```cpp
const uint32_t my_chunk_start = core_idx_in_group * chunks_per_worker;
const uint32_t my_chunk_end =
    min((core_idx_in_group + 1) * chunks_per_worker, valid_k_chunks);
for (uint32_t k_chunk = my_chunk_start; k_chunk < my_chunk_end; ++k_chunk) {
    ...read K/V chunk k_chunk...
}
```
The reducer also reads its own slice, plus stays alive after to pull
partial state from neighbors.

### D. Compute changes
- Iterate the chunk loop only over `[my_chunk_start, my_chunk_end)`.
- The first chunk in the slice is treated as `k_chunk == 0` for the lazy
  softmax correction (its max/sum/out initialise the running state — no
  prior to merge with).
- After the loop:
  - **Workers** (core_idx_in_group != 0): pack final `(max_w, sum_w, out_w)`
    into a designated L1 location (e.g. cb_partial_max, cb_partial_sum,
    cb_partial_out — three new CBs). Signal the reducer's semaphore.
    Exit kernel.
  - **Reducer** (core_idx_in_group == 0):
    1. Wait on semaphore (count == cores_per_head - 1).
    2. For each w in 1..K-1:
       - Pull (w_max, w_sum, w_out) from worker `w`'s L1 via
         `noc_async_read` into local CB tiles.
       - Run online-softmax merge against running (prev_max, prev_sum,
         prev_out).
    3. After all merges: `out = prev_out / prev_sum`, pack to cb_out.
  - Existing writer logic on the reducer writes cb_out to global output.

### E. Synchronization
Per (B, NQH) group:
- Allocate one semaphore (`reducer_done_sem`) at compile time via
  `tt_metal::CreateSemaphore`.
- Workers do `noc_semaphore_inc(reducer_addr, 1)` after writing their
  partial state.
- Reducer does `noc_semaphore_wait(local_sem_addr, K - 1)`.

### F. New CBs
Three new CBs per core for partial state:
- `cb_partial_max` (Sq_chunk_t tiles, BF16) — running max
- `cb_partial_sum` (Sq_chunk_t tiles, BF16) — running sum
- `cb_partial_out` (Sq_chunk_t * vDHt tiles, BF16) — running out

Reducer also needs *neighbor* CBs to receive remote partials. Either
(option A) allocate K-1 extra triplets and round-robin, or (option B)
reuse one triplet sequentially (simpler).

---

## Risks & open questions

1. **L1 budget**: With K=14 workers and Sq_chunk_t=1, vDHt=4, partial state
   is 6 tiles per worker (1+1+4) × 2 bytes/elem × 1024 elems/tile = 12 KB
   per worker. Reducer holds at most `K * 12 KB ≈ 170 KB` if we keep all
   partials in L1 for parallel merge. L1 per core is 1.5 MB so plenty of
   headroom.

2. **Page table sharing**: each worker reads its own slice of the paged
   cache. Page table is replicated — fine.

3. **Pre-rescaled mode**: existing fused-SDPA path already uses
   `sdpa_standard` for pre_rescaled (no chunk loop). Skip 2A there — only
   apply the new path when `pre_rescaled == false`. Add a `cores_per_head`
   check that asserts 1 when pre_rescaled.

4. **Dynamic chunk count**: `valid_k_chunks` is data-dependent (cur_pos).
   `chunks_per_worker` must be computed at runtime. The semaphore count is
   actually `min(K, valid_k_chunks)` because workers with no chunks should
   exit early (matches sdpa_decode's `has_local_data` skip).

5. **Mesh / multi-device**: each device runs independently — 2A is per-device
   parallelism only. No cross-device communication needed. (T3K just gives
   you fewer NQH per device, so K can be larger.)

---

## Implementation phases

**Phase 1 — chunk-range scaffolding (~1 day)**
- Add `core_idx_in_group` and `cores_per_head` runtime args to
  reader + compute. Default to (0, 1) — no behavior change.
- Verify `test_paged_partial_cache.py` + `test_mesh_fused_sdpa.py` + e2e.
- Add `num_cores_per_head` attribute (default 1) to device op + binding.

**Phase 2 — split + linear reduce (~3-5 days)**
- Implement worker/reducer split.
- Allocate partial-state CBs.
- Add semaphores + NoC-pull merge in reducer.
- Verify with `cores_per_head=2` (smallest non-trivial), then 4, 8, 14.

**Phase 3 — perf tuning (~2-3 days)**
- Run `bench_seqlen_sweep.py` at all power-of-2 seqs with
  `cores_per_head=14` on T3K.
- Compare to BFP8 baseline. Goal: T3K e2e `< 1.4× of baseline` at 32K.

**Phase 4 — tree reduce (optional, ~3-5 days)**
- If linear-reduce overhead grows for K=14, switch to `log2(K)` rounds of
  pairwise merges. Mirrors sdpa_decode tree reduction.

**Phase 5 — cleanup + docs (~1 day)**
- Update `PLAN.md` with final perf numbers.
- Document `num_cores_per_head` knob in the operator help string.
- Update bench script to plot 1-vs-K speedup curve.

---

## Files affected (rough estimate)

- `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/device/sdpa_tq_device_operation.{cpp,hpp}` — attribute
- `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/device/sdpa_tq_program_factory.cpp` — work distribution + per-core args + semaphores
- `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/sdpa_tq_pybind.cpp` — binding for new kwarg
- `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/dataflow/reader_tq_decode.cpp` — chunk-range slice
- `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/compute/sdpa_tq_decode.cpp` — chunk-range loop + reducer logic
- `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/dataflow/writer_tq_decode.cpp` — only reducer writes
- `turbo_quant/ttnn_integration.py` — pass `num_cores_per_head` through
- `turbo_quant/eval_e2e.py` — optional CLI flag
