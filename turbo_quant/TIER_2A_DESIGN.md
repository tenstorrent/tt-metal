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

**Phase 1 — chunk-range scaffolding (✅ DONE — commit a6d2903)**
- Compute kernel now uses `(k_chunk_start_for_core, k_chunk_end_for_core)`
  as loop bounds instead of `(0, valid_k_chunks)`.
- `core_idx_in_group` and `cores_per_head_runtime` are still `constexpr (0, 1)`.
- Lazy-correction guard updated to `k_chunk > k_chunk_start_for_core`.
- Validated: cos=0.9969 (unchanged), T3K e2e 26.9 ms/tok (unchanged).

**Phase 2.1 — runtime arg plumbing (✅ DONE — commit 9d2fece)**
- Compute kernel reads `core_idx_in_group_arg` / `cores_per_head_arg`
  from runtime arg slots [7] / [8] (repurposed unused slots, not appended).
- Program factory sends (0, 1) per core — same behaviour as before.
- Root cause of earlier wedge: forgot to `./build_metal.sh` after editing
  the program factory. Reused-slot strategy plus rebuild discipline
  unblocked it.

**Phase 2.2a — `num_cores_per_head` attribute end-to-end (✅ DONE — commit d062460)**
- Added `num_cores_per_head` to `operation_attributes_t` (default 1).
- Threaded through nanobind / cpp wrapper / launch / program factory.
- Program factory clamps to `num_cores / (B * NQH)`.

**Phase 2.2b — K cores per (B, NQH) tuple (✅ DONE — commit ecc83d4)**
- Per-core runtime-args loop: `group_id = i / cores_per_head`,
  `core_idx_in_group = i % cores_per_head`.
- Cores at idx > 0 within a group get an empty (batch, head) range so
  their kernel main exits early. Runtime `cores_per_head_arg` sent to
  the compute kernel is forced to 1 until the reduce phase lands.
- New test `test_2A_cores_per_head.py` confirms K=1, 2, 4 produce
  bit-identical output (max|diff| = 0) on a populated paged cache.

**Phase 2.3 step 1 — partial-state CBs + semaphore (✅ DONE — commit e74354b)**
- Allocated six BF16 CBs (c_18..c_23) — three for worker pack, three for
  reducer remote-pull. ~12 KB per core.
- Allocated one program-wide reducer semaphore via `tt_metal::CreateSemaphore`.
- Wired semaphore_id into compute kernel runtime arg slot [9].

**Phase 2.3 step 2a — compute kernel reads new args (✅ DONE — commit a4a4c7d)**
- Kernel now reads `reducer_semaphore_id` from arg [9].
- Derives `is_reducer = (core_idx_in_group_arg == 0)`.
- Declares constexpr CB indices for cb_partial_max/sum/out and
  cb_remote_max/sum/out. All [[maybe_unused]] for now.
- K=1, 2, 4 still bit-identical.

**Phase 2.3 step 2b — reader chunk slicing (✅ DONE — commit 34850b7)**
- Reader reads `core_idx_in_group_arg` / `cores_per_head_arg` from
  new runtime arg slots and slices the chunk loop accordingly.
- Program factory passes the same forced (0, 1) values.

**Phase 2.3 step 2c — compute kernel worker pack-and-skip (✅ DONE — commit 60c5e2d)**
- After matmul_reduce, when `cores_per_head_arg > 1 && !is_reducer`,
  the compute kernel copies the running (max, sum, out) tile-by-tile
  into cb_partial_max/sum/out and `continue`s — skipping the dilution
  correction, recip, and normalize.
- Dead code at K=1; K=1, 2, 4 still bit-identical.

**Phase 2.3 step 3 — writer NoC-send / wait-and-push (✅ DONE — commit 4d706be)**
- Worker writer NoC-async-writes cb_partial_max/sum/out into the
  reducer's cb_remote_max/sum/out L1 slots, then noc_semaphore_inc's
  the reducer.
- Reducer writer noc_semaphore_wait's for K-1 increments, resets the
  sema, then cb_push_back's cb_remote_* so the compute merge loop
  unblocks.
- Program factory computes the reducer's physical NoC (x, y) via
  `device->worker_core_from_logical_core(group_id * cores_per_head)`
  and passes (x, y) + semaphore_id to every core's writer.
- Both branches dead at K=1; bit-identical confirmed.

**Phase 2.3 step 4 — reducer compute wait-and-merge (NEXT)**

Concrete steps:

1. **Program factory — semaphores + partial-state CBs**
   - One `tt_metal::CreateSemaphore(program, all_cores, 0)` per program.
     Returns the semaphore's L1 address; pass to all cores.
   - Three new BF16 CBs per core for partial state:
     - `cb_partial_max` (Sq_chunk_t tiles)
     - `cb_partial_sum` (Sq_chunk_t tiles)
     - `cb_partial_out` (Sq_chunk_t * vDHt tiles)
     These are local-only on workers (data sits in L1); the reducer reads
     remote copies via NoC into matching local CBs (`cb_remote_*`).
   - Add per-core runtime args at slots [10..]:
     - `is_reducer` (1 if core_idx_in_group == 0)
     - `num_workers` (cores_per_head, set to actual K not the forced 1)
     - For reducer: NoC physical (x, y) of the K-1 worker peers.

2. **Compute kernel — worker / reducer split**
   - Worker (idx > 0): after the existing chunk loop, instead of
     `recip_block_inplace + mul_block_bcast_cols → cb_out`, copy the
     final (alias_prev_max, alias_prev_sum, alias_mm2_prev_out) into
     cb_partial_max/sum/out. Then `noc_semaphore_inc(reducer_addr, 1)`
     and exit.
   - Reducer (idx == 0): after its own chunk loop completes,
     `noc_semaphore_wait(local_sem_addr, num_workers - 1)`. Loop over
     workers 1..K-1: `noc_async_read` their cb_partial_* tiles into
     local cb_remote_*; perform the standard online-softmax merge:
     ```
     new_max  = max(prev_max, w_max)
     alpha    = exp((prev_max - new_max) * scale)
     beta     = exp((w_max - new_max) * scale)
     new_sum  = alpha * prev_sum + beta * w_sum
     new_out  = alpha * prev_out + beta * w_out
     ```
     After all merges: `out = new_out / new_sum` and pack to cb_out.

3. **Writer kernel — gate on is_reducer**
   - Currently the writer iterates `[batch_start, batch_end) ×
     [head_start, head_end)`. With idx > 0 cores already getting empty
     ranges, the writer for them is a no-op — no change needed unless
     we want to assert/log it.

4. **Program factory — flip the kernel `cores_per_head_arg`**
   - Currently forced to 1. Once 1+2 are working, send the real
     `cores_per_head` so the chunk-loop bounds split. Verify K=1 still
     correct, then K=2.

5. **L1 budget check**
   - Three new CBs × (1 + 1 + 4 = 6) tiles × 2 bytes/elem × 1024
     elems/tile ≈ 12 KB per core. Reducer needs another set for
     remote-pull → 24 KB total. L1 is 1.5 MB so plenty of headroom.

**Validation strategy**
- After each step: `test_2A_cores_per_head.py` at K=1 must pass
  (identical output).
- After step 4: `test_2A_cores_per_head.py` at K=2 must pass with cos
  > 0.999 vs K=1 (bit-identity is unrealistic with the merged path's
  re-ordered ops).
- Then K=4, K=8, K=14 (T3K full grid).
- Once stable: re-run `bench_seqlen_sweep.py` on T3K with K=14 and
  measure the per-call latency gain at long context.

### Open issues for step 4 (reducer merge)

1. **Multiple workers all write to the SAME reducer slot.** Current writer
   has all K-1 workers `noc_async_write`ing to the reducer's
   `cb_remote_max[0]` — race condition. Fix options:
   - **(a)** Allocate `cb_remote_*` with `cores_per_head` slots; worker w
     writes to slot w (offset `w * tile_bytes * Sq_chunk_t`). Reducer
     reads slots 1..K-1 sequentially. Simpler L1, more memory.
   - **(b)** Sequential: reducer signals worker_w "your turn", worker_w
     sends, etc. Less L1 but more latency.
   - Pick (a). Need to plumb `cores_per_head` into the cb allocation
     (already available at program-build time as the clamped attribute).

2. **Online softmax merge math is symmetric.** The chunk-loop's lazy
   correction assumes `cur_max ≥ prev_max` because reduce_c with
   eltwise_max enforces that. Cross-core merge has no such guarantee —
   the peer's max could be larger or smaller. Need both
   `exp_max_diff_self` and `exp_max_diff_peer` (mirror sdpa_flash_decode's
   tree-reduction code).

3. **Workers don't apply dilution correction.** Each worker only sees
   `cur_pos % chunks_per_worker` real positions in its slice (or none, if
   its slice is past cur_pos). Currently the dilution correction uses
   `valid_k_chunks * k_chunk_size_tokens` for total iterated and
   `cur_pos + 1` for real count — which is the *global* count.
   - The reducer must apply the dilution correction with global counts
     after merging all workers. Workers should NOT apply it themselves
     (they'd double-count zeros).
   - Workers also need to skip the matmul_reduce — actually no, the
     matmul_reduce just consolidates the per-row sum, it's a no-op for
     a properly-rowmaxed sum. Both worker and reducer can do it; merge
     happens after.

### Step 4 plan

- Update CB allocation: `cb_remote_max/sum` size = `cores_per_head * Sq_chunk_t`,
  `cb_remote_out` size = `cores_per_head * out_chunk_tiles`.
- Update worker writer: write to offset `core_idx_in_group * tile_bytes * count`.
- Update reducer writer: cb_push_back per-slot with proper sequence.
- Implement the merge loop in compute kernel after own `matmul_reduce`:
  - For each peer w in 1..K-1:
    - cb_wait_front for that peer's slot (or single CB if reducer reads
      one at a time)
    - Compute `new_max = max(prev_max, cb_remote_max[peer])` via
      `max_block` or `reduce_c<MAX>`
    - Compute `exp_max_diff_self = exp((prev_max - new_max) * scale)`
    - Compute `exp_max_diff_peer = exp((cb_remote_max[peer] - new_max) * scale)`
    - `prev_sum = exp_max_diff_self * prev_sum + exp_max_diff_peer * cb_remote_sum[peer]`
    - `prev_out = exp_max_diff_self * prev_out + exp_max_diff_peer * cb_remote_out[peer]`
    - `prev_max = new_max`
- Then existing dilution correction + recip + normalize → output.

### Pickup checklist for next session

Resume at Phase 2.3 step 4 (above). The next concrete edits are:

1. **Program factory: stop forcing empty (batch, head) range for idx > 0**
   - Currently `if (core_idx_in_group != 0) { batch_end = batch_start;
     head_end = head_start; }`. Remove this so all workers in a group
     share the same (batch, head) tuple.
   - Stop forcing `kernel_cores_per_head = 1`; pass actual `cores_per_head`.
2. **Compute kernel: insert worker branch at end of (nb, nq) iteration**
   ```cpp
   if (!is_reducer) {
       matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);
       // Copy alias_prev_max → cb_partial_max via tile_regs path
       // Copy alias_prev_sum → cb_partial_sum
       // Copy alias_mm2_prev_out → cb_partial_out
       cb_pop_front(alias_prev_max, Sq_chunk_t);
       cb_pop_front(cb_q_in, q_chunk_tiles);
       continue;  // skip the recip + normalize + dilution-correction
   }
   ```
3. **Writer kernel: add worker NoC-send path**
   - For each worker (idx > 0): `cb_wait_front(cb_partial_*)`,
     `noc_async_write` each tile to reducer's L1 at the matching
     cb_remote_* address, `noc_semaphore_inc` reducer.
   - Reducer NoC physical (x, y) passed as new runtime args.
4. **Reducer compute path**
   - After own chunk loop + matmul_reduce, before recip:
     `cb_wait_front(cb_remote_*)` for each peer (driven by sema-incremented
     by writer), then perform online-softmax merge:
     ```
     new_max = max(prev_max, w_max)
     alpha   = exp((prev_max - new_max) * scale)
     beta    = exp((w_max - new_max) * scale)
     prev_sum = alpha * prev_sum + beta * w_sum
     prev_out = alpha * prev_out + beta * w_out
     prev_max = new_max
     ```
     Then apply the existing dilution correction with the *global*
     `valid_k_chunks` and finalize as today.
5. **Validate**: `test_2A_cores_per_head.py` at K=2, then K=4, K=14.
   Then `bench_seqlen_sweep.py` to capture the speedup at long
   context.

Files to touch:
- `sdpa_tq_program_factory.cpp` (steps 1, 3a — add reducer NoC coords runtime args)
- `kernels/compute/sdpa_tq_decode.cpp` (step 2, 4)
- `kernels/dataflow/writer_tq_decode.cpp` (step 3)
- Tried promoting `core_idx_in_group` and `cores_per_head_runtime` from
  `constexpr (0, 1)` to runtime args [10] and [11]. Program factory set
  them to (0, 1) per-core — no logical change.
- **Result:** kernel hung with `Read unexpected run_mailbox value` error.
  Device required `tt-smi -r` to recover; subsequent runs of the
  identical reverted code also timed out, suggesting the failed run left
  the device in a wedged state that the test runner could not recover
  from inside the same Python process.
- **Hypothesis on why it hung:** the original program factory passed 10
  runtime args; my version passed 12. The device-op's program cache may
  have keyed the existing program on the old arg count, and the new args
  were either not propagated to the right slots or trampled cache state.
  Alternatively the kernel's stack/mailbox layout shifted in a way that
  trips an alignment check.
- **Next session must:**
  1. Reset the device with `tt-smi -r`.
  2. Re-add the runtime args, but bisect carefully: first add only one
     extra arg (e.g. just `cores_per_head_arg`) and re-run; if that
     works, add the second.
  3. If still hangs, check whether `override_runtime_arguments` needs to
     also be updated to reflect the new arg layout (currently it only
     touches reader_args; compute_args is set once at create and not
     overridden, which should be fine but may interact badly with the
     program cache).
  4. Try forcing a fresh program by clearing
     `~/.cache/tt-metal-cache/*/kernels/sdpa_tq_decode` before running.

**Phase 2.2-2.5 — actual split + reduce (planned)**
- See "Algorithm" and "Plumbing required" sections above. ~3-5 days
  combined.

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
