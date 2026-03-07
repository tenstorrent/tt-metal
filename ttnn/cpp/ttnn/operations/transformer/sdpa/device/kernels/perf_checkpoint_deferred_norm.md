# SDPA Streaming v2 Ring: Deferred Normalization + Row-by-Row Pipeline

Date: 2026-03-07
Branch: `dnijemcevic/sdpa_norm_integration`

## Progress Summary

### Completed and staged
1. **Deferred normalization** — replaces sigmoid merge with exp-rescaling (prior commits)
2. **Writer prefetch** — after writing Q_N's accumulators, immediately prefetch Q_{N+1}'s from DRAM
3. **K0 is_first=true + deferred merge** — K chunk 0 runs fresh (no SALAD needed); DRAM prev_out merged afterward via standalone `merge_prev_accumulators`. Entire K0 compute time overlaps with writer's DRAM read. Fallback to old path when total_valid_kv==1.
4. **Row-by-row save** — `copy_block_row_by_row` replaces bulk `copy_block` in save path. Per-group `cb_wait_front`/`cb_pop_front` on input, per-group `cb_reserve_back` on output. Frees cb_out incrementally.
5. **Row-by-row DRAM write** — `write_output_row_by_row` writes output to DRAM in subblock_h-row groups (both non-last and last ring_iter)
6. **Row-by-row DRAM read** — `read_prev_accumulators_prefetch` reads prev_out row-by-row, max/sum all at once
7. **Row-by-row final write** — `write_output_row_by_row` for `is_last_ring_iter` too (was `write_block`)
8. **Code comments** — naming conventions (Nt, global, local, joint), struct/function docs, CT/RT arg annotations

9. **Direct output from last K chunk** — `output_row_direct` lambda streams finalized rows from cur.out to cb_out during SALAD on the last K chunk, eliminating post-loop `copy_block_row_by_row`. Three bugs fixed:
   - **Fired on intermediate K chunks**: `direct_output_cb` was passed to every K chunk; now only passed for the last valid K chunk of non-last ring iterations (`KV_chunks_processed == total_valid_kv && !is_last_ring_iter`).
   - **`!is_last_iter` guard missing**: On the last ring iteration's last K chunk, both `normalize_row` (inside `salad_correct_row`) and `output_row_direct` would fire on the same cur.out row. Added `&& !is_last_iter` to the `output_row_direct` calls.
   - **Shared `pushed_rows` counter**: `output_row_direct` pushes/pops cur.out but NOT cur.sum. Using the same `pushed_rows` counter for both caused SALAD to write cur.sum at wrong offsets. Split into `pushed_rows` (normalize_row: both) + `out_only_pushed` (output_row_direct: cur.out only). `salad_correct_row` now takes separate `w_out`/`w_sum` write offsets.

### PCC Results (all 36 configs pass with direct output enabled)
- Ring joint SDPA: 2 seq_lens × 3 Q chunks × 3 K chunks = 18 configs ✓
- Standard SDPA: 2 seq_lens × 3 Q chunks × 3 K chunks = 18 configs ✓

### Performance (CCLs commented out)
| Config | Original | Current | Delta |
|--------|----------|---------|-------|
| s=8544 q288/k512 | 8.566ms / 63.1% | 8.575ms / 63.1% | noise |
| s=2240 q224/k512 | 0.704ms / 52.8% | 0.708ms / 52.5% | noise |

Performance neutral with CCLs disabled — expected since there's no real cross-device latency to hide.

## Architecture

### Ring SDPA data flow
- K/V data rotates across devices via CCL (ring topology)
- Loop order: `ring_iter → q_chunk → k_chunk`
- Each core owns private Q chunks; no cross-core Q sharing
- K/V is transient — all Q chunks must process current K/V shard before it rotates

### Standard online softmax (single device)
Standard streaming attention processes K chunks sequentially, maintaining running
statistics. For each K chunk, the inner loop (`sdpa_inner_loop_step`) does:

1. **Phase 1 — Q@KT + running max**: Matmul Q×K^T row by row into `cb_qkt_im`,
   computing running row-max into `cur.max` (element-wise max with `prev.max`
   when `!is_first`).

2. **Phase 2 — Softmax drain + V matmul + SALAD**: Processes rows in a pipeline:
   - **Row N V matmul**: `softmax(QK^T) × V` → writes to `cur.out` (reserved space).
   - **Row N-1 SALAD correction** (when `!is_first`): rescale previous accumulator
     to align with the new max: `cur.out += prev.out × exp(prev_max - cur_max)`,
     `cur.sum += prev.sum × exp(prev_max - cur_max)`. Uses L1-accumulate
     (`pack_tile<true>` with `l1_acc=1`) to add onto cur.out/cur.sum in-place.
   - **Row N-1 normalization** (when `is_last`): `output = cur.out / cur.sum`
     via streaming matmul-reduce + reciprocal + broadcast-multiply.

3. **Bulk push**: Push `cur.max`, `cur.sum`, `cur.out` to make them available
   for the next K chunk's SALAD (as prev) or for the final output.

The pipeline overlaps the V matmul for row N with the SALAD correction for
row N-1 within the same K chunk, then drains the final row after the loop.

### Deferred normalization (replaces sigmoid merge)
In ring SDPA, each ring iteration processes a remote device's K/V shard. The old
approach merged results across ring iterations using a sigmoid-based LSE blend
(~11 tile operations per iteration). Deferred normalization replaces this with:

- **Across K chunks (within a ring iteration)**: standard SALAD rescaling
  `exp(old_max - new_max)` — same as single-device.
- **Across ring iterations**: the SAME SALAD rescaling, treating each ring
  iteration's K/V shard as additional K chunks. No separate merge pass.
- **Single normalization**: `output / sum` happens once at the very end (last K
  chunk of last ring iteration), not after every ring iteration.

This works because online softmax's rescaling is associative: it doesn't matter
whether the K chunks come from the local device or a remote one.

**Two paths based on q_per_core:**
- **q_per_core == 1**: Accumulators (max, sum, out) persist in L1 across ring
  iterations using ping-pong CBs. No DRAM round-trip. The `RingAccumulatorState`
  struct tracks which CB pair holds current values and whether `is_first` globally.
- **q_per_core > 1**: Multiple Q chunks share the same CB pair, so accumulators
  must be saved to DRAM after each Q chunk and restored before the next ring
  iteration. The writer handles DRAM I/O; compute sees CBs.

### K0 deferred merge (multi-Q, ring_iter > 0)
When restoring from DRAM, the naive approach blocks compute waiting for the full
`prev_out` read. The deferred merge optimizes this:

1. **Restore max/sum only** (small, fast) from DRAM.
2. **K0 runs with `is_first=true`** — fresh compute, no SALAD needed, no prev_out
   dependency. K0's compute overlaps with the writer's DRAM read of prev_out.
3. **`merge_prev_accumulators`**: after K0 completes, a standalone SALAD pass
   blends the DRAM prev_out with K0's result: `K0.out += DRAM_out × exp(DRAM_max - K0_max)`.
4. **K1+ proceed normally** with standard SALAD.

Fallback: when `total_valid_kv == 1` (only one K chunk), K0 is both first and
last — can't defer. Loads prev_out upfront and runs with `is_first=false`.

### Row-by-row direct output pipeline
After the K-chunk loop, accumulated output tiles sit in the accumulator CB
(cur.out, in reserved state). The old approach bulk-pushed all tiles then copied
them to cb_out via `copy_block_row_by_row` post-loop. Direct output eliminates
this extra copy by streaming tiles to cb_out **during** the last K chunk's SALAD:

```
Last K chunk, SALAD pipeline (row by row):
  row N:   V matmul → cur.out[N]
  row N-1: SALAD correction → cur.out[N-1] is finalized
           output_row_direct: push cur.out[N-1], copy to cb_out, pop
  ...
  drain:   SALAD correction → cur.out[last] is finalized
           output_row_direct: push cur.out[last], copy to cb_out, pop
```

Each `output_row_direct` call:
1. `cb_push_back(cur.out, sbh × vDHt)` — make the SALAD-corrected row readable
2. `copy_tile` through DST → `pack_tile` to `direct_output_cb` (cb_out)
3. `cb_pop_front(cur.out, sbh × vDHt)` — free space in cur.out

This shifts cur.out's wr_ptr, so subsequent V matmul writes use adjusted offsets
(`w_q = q_subblock - pushed_rows - out_only_pushed`).

**Key constraint — dual pushed counters**: `output_row_direct` pushes/pops
cur.out but NOT cur.sum (sum is bulk-pushed after the K chunk for DRAM save).
This means cur.out and cur.sum wr_ptrs diverge. The SALAD correction must use
separate write offsets:
- `w_out = salad_row - pushed_rows - out_only_pushed` for cur.out
- `w_sum = salad_row - pushed_rows` for cur.sum

Where `pushed_rows` counts normalize_row calls (pushes both) and `out_only_pushed`
counts output_row_direct calls (pushes cur.out only). These are mutually exclusive:
normalize fires on the last ring iteration; direct output fires on non-last.

**When direct output fires**: only on the last valid K chunk of non-last ring
iterations (`q_per_core > 1 && !is_last_ring_iter && KV_chunks_processed == total_valid_kv`).
Intermediate K chunks must keep their accumulator intact for the next SALAD.
The last ring iteration uses normalize_row instead (output → cb_normalized_out → cb_out).

### Current data flow (multi-Q, ring_iter > 0)

**Writer per Q chunk:**
```
if first Q: read_prev_accumulators_prefetch(Q0)  // max/sum all at once, prev_out row-by-row
// wait for compute to finish Q chunk
if is_last_ring_iter: write_output_row_by_row(final output)
else: write_output_row_by_row(accum output) + write_max_and_sum
if has_next_q: read_prev_accumulators_prefetch(Q_{N+1})  // prefetch overlaps with K0 compute
```

**Compute per Q chunk (ring_iter > 0, total_valid_kv > 1):**
```
copy max/sum from DRAM upfront (small)
K0: is_first=true, skip_cur_push=true  // fresh compute, no SALAD
swap prev/cur
merge_prev_accumulators(DRAM prev_out + K0 result)  // standalone SALAD pass
K1..K_{N-2}: is_first=false, normal SALAD
K_{N-1} (last): is_first=false, SALAD + output_row_direct per row → cb_out
  (on last ring_iter: SALAD + normalize_row per row → cb_normalized_out)
save: copy max/sum to DRAM writer CBs
  (output already in cb_out from direct output; skip copy_block_row_by_row)
```

**Compute per Q chunk (ring_iter > 0, total_valid_kv == 1, fallback):**
```
copy max/sum AND prev_out from DRAM upfront (blocking)
K0: is_first=false, normal SALAD with DRAM accumulators + output_row_direct
```

### Key files

| File | Role |
|------|------|
| `compute_streaming.hpp` | `sdpa_inner_loop_step` (+ `skip_cur_push`, `direct_output_cb` params), `sdpa_ring_v2`, `merge_prev_accumulators`, `copy_block_row_by_row`, `output_row_direct` lambda |
| `ring_joint_sdpa.cpp` | Compute kernel main — dispatches to streaming v2 or old path |
| `ring_joint_writer.cpp` | Writer kernel — `write_output_row_by_row`, `read_prev_accumulators_prefetch`, `write_max_and_sum`, `QChunkAddr` helper |
| `ring_joint_sdpa_program_factory.cpp` | Host-side CB allocation, compile-time args |
| `compute_common.hpp` | `copy_block` (original bulk version, still used for max/sum) |

### New functions added

**compute_streaming.hpp:**
- `copy_block_row_by_row(in_cb, out_cb, Sq_chunk_t, vDHt, sbh)` — per-group wait/pop on input, per-group reserve on output. Avoids holding input CB across blocking output reserve.
- `merge_prev_accumulators<...>(q_prev, q_cur, cb_prev_out, Sq_chunk_t)` — standalone SALAD pass: `q_prev.out += DRAM_out * exp(DRAM_max - K0_max)`, pushes merged result, pops DRAM data. cb_exp_max_diff tiles accumulated across rows, popped all at end.
- `output_row_direct` lambda (inside `sdpa_inner_loop_step`) — push cur.out row, copy through DST to direct_output_cb, pop cur.out. Uses `out_only_pushed` counter (separate from `pushed_rows` which tracks normalize_row). Fires only on last K chunk of non-last ring iterations.

**ring_joint_writer.cpp:**
- `write_output_row_by_row(gen, slice, end, cb, bytes, Sq, DHt, sbh)` — per-group: wait, write tiles to DRAM, barrier, pop
- `read_prev_accumulators_prefetch(gen, stats_writer, ..., sbh)` — max/sum all at once, prev_out row-by-row push
- `write_max_and_sum(stats_writer, ..., cb_max_out, cb_sum_out, bytes)` — extracted from old `write_accumulators`
- `QChunkAddr` struct + `compute_q_chunk_addr<...>(global_q, ring_id)` — eliminates duplicate addressing between main loop and prefetch

### Writer compile-time args
```
Index 22: use_deferred_norm
Index 23: subblock_h (added for row-by-row sizing)
Index 24+: TensorAccessorArgs for output, joint_output, LSE tensors
```

### CB layout (unchanged)

| CB | Index | Purpose |
|----|-------|---------|
| cb_q_in | c_0 | Q input |
| cb_k_in | c_1 | K input |
| cb_v_in | c_2 | V input |
| cb_mask_in | c_3 | Lightweight mask tiles |
| cb_scale_in | c_4 | Scale scalar |
| cb_identity_scale_in | c_5 | Identity scale for reduce |
| cb_lse_in | c_6 | Previous max from DRAM |
| cb_prev_out | c_7 | Previous output from DRAM |
| cb_col_identity | c_8 | Column identity for matmul_reduce |
| cb_recip_scratch | c_9 | 1-tile scratch for normalize_row_streaming |
| cb_sum_out | c_10 | Sum save to DRAM |
| cb_sum_in | c_11 | Sum restore from DRAM |
| cb_out | c_16 | Output to DRAM / normalized output |
| cb_lse_out | c_17 | Max save to DRAM |
| cb_qk_im | c_24 | Q@KT intermediate |
| cb_out_im_A | c_25 | Output accumulator ping |
| cb_out_im_B | c_26 | Output accumulator pong |
| cb_max_A/B | c_27/28 | Max accumulator ping/pong |
| cb_sum_A/B | c_29/30 | Sum accumulator ping/pong |
| cb_exp_max_diff | c_31 | exp(prev_max - cur_max) scratch |

### Subblock sizes for key configs
`determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size=8)`:
- q288 (Sq=9): sbh=1, sbw=8 (9 is odd → only 1 divides it and fits ≤2)
- q224 (Sq=7): sbh=1, sbw=8 (same reason)
- q256 (Sq=8): sbh=2, sbw=4

## Debugging Methodology

### Rules (CRITICAL — follow these exactly)

1. **15-second timeout, no exceptions.** Any test taking >15s on warm cache is hung. Do NOT extend timeouts. Do NOT re-run hoping it will pass. Do NOT blame cold cache or space radiation.

2. **tt-triage immediately on hang.** Do NOT kill the hung process first. Run triage while the test is still stuck:
   ```bash
   python3 tools/tt-triage.py > /tmp/triage.txt 2>&1
   grep -E "Kernel Name|#0 |#1 |#2 " /tmp/triage.txt | head -30
   ```
   The callstacks show exactly where each RISC-V core is blocked.

3. **Always nuke cache after build.** `rm -rf ~/.cache/tt-metal-cache` after every `./build_metal.sh`. This is hygiene, not debugging. Never wonder "is this stale binary?"

4. **One config at a time.** Edit the .py file to restrict to a single Q/K chunk combo:
   ```python
   Q_CHUNK_SIZES = [288]
   K_CHUNK_SIZES = [512]
   ```

5. **DPRINT for targeted debugging.** When triage shows the stuck location but root cause isn't obvious:
   ```bash
   TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=TR2 pytest ... -s
   ```
   - `DPRINT_CORES=0,0` limits to logical core (0,0) on all devices
   - `DPRINT_RISCVS=TR2` limits to trisc2 (PACK risc, where cb_reserve_back blocks)
   - `DPRINT_RISCVS=BR` for BRISC (writer/reader), `NC` for NCRISC
   - Use `DPRINT_PACK(DPRINT << "msg" << ENDL())` in compute kernels
   - Include `"api/debug/dprint.h"` — REMOVE after debugging (causes JIT compile errors if left)

6. **Read triage callstacks carefully.** Key patterns:
   - `cb_reserve_back` in compute → CB full, consumer (writer) hasn't popped
   - `cb_wait_front` in writer → CB empty, producer (compute) hasn't pushed
   - Both stuck → deadlock: compute holds one CB while waiting on another

7. **Cold cache takes ~25s.** First run after `rm -rf ~/.cache/tt-metal-cache` JIT-compiles kernels. Triage during this shows idle cores (no kernel dispatched). Subsequent runs take ~7-10s. Don't confuse cold cache with a hang — triage will tell you.

### Bug patterns encountered

**Stale flag after disabling a feature:**
Disabled `direct_output_cb` (set to 0) but forgot the `used_direct_output` flag was set by a SEPARATE condition that didn't check `direct_output_cb`. The flag skipped the save path (`copy_block_row_by_row`), so cb_out never got data → writer deadlocked waiting on empty cb_out. Fix: tie the flag to the actual `direct_output_cb` value, or remove the flag-set when direct output is disabled.

**Cumulative CB waits with per-iteration pops:**
`cb_exp_max_diff` uses cumulative waits inside `mul_block_bcast_cols_acc`: `cb_wait_front(cb, (row+1)*sbh)`. Initial `merge_prev_accumulators` code popped per row group → cumulative wait at group 1 expected 2 tiles but only 1 available (previous was popped). Fix: accumulate pushes across all groups, pop ALL at the end (`cb_pop_front(cb_exp_max_diff, Sq_chunk_t)`).

**Holding input CB across blocking output reserve:**
Initial `copy_block_row_by_row` did `cb_wait_front(in_cb, ALL)` upfront then per-group `cb_reserve_back(out_cb)`. If out_cb was full (writer slow), the reserve blocked while in_cb (accumulator CB) was held. Next Q chunk's `cb_reserve_back(cur.out)` on the SAME accumulator CB deadlocked. Fix: wait and pop per group on the input side too — never hold the input across a potentially-blocking output reserve.

**output_row_direct triple bug (fired wrong K chunk + no is_last guard + shared pushed_rows):**
Three independent issues combined to produce garbage (PCC 0.008, RMSE inf):
1. `direct_output_cb` was passed to ALL K chunk iterations; should only be non-zero for the LAST valid K chunk (intermediate K chunks need the accumulator for the next SALAD).
2. On the last ring iteration's last K chunk, both `normalize_row` and `output_row_direct` fired on the same cur.out row (double-processing). Added `&& !is_last_iter` guard.
3. `output_row_direct` pushes/pops cur.out but NOT cur.sum. Using a single `pushed_rows` counter caused SALAD to write cur.sum at wrong offsets (row overwrites). Split into `pushed_rows` (normalize_row) + `out_only_pushed` (output_row_direct), with separate `w_out`/`w_sum` in `salad_correct_row`.

## Next Steps

- **Performance measurement**: Re-measure with CCLs enabled to see if direct output provides actual latency benefit (overlapping output copy with writer DRAM operations).
- **Non-uniform dataformat**: `output_row_direct` may need `pack_reconfig_data_format(direct_output_cb)` for non-bf16 formats. Currently only bf16 (uniform_dataformat=true) is tested.

## How to Run Tests

### Prerequisites
```bash
source python_env/bin/activate
./build_metal.sh   # after kernel changes
rm -rf ~/.cache/tt-metal-cache  # always clear after build

# tt-triage (install once):
/localdev/dnijemcevic/tt-metal/scripts/install_debugger.sh
uv pip install -r /localdev/dnijemcevic/tt-metal/tools/triage/requirements.txt
```

### CCL management
```
ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/
  ring_attention_all_gather_reader.cpp
  ring_attention_all_gather_writer.cpp
```
Comment out (for perf): `git show 5cb97900bb:<path> > <path>`
Restore (for PCC):     `git show 5cb97900bb^:<path> > <path>`

### Test files
```
Ring:     tests/.../test_ring_joint_attention_scaled_dot_product_attention_sprint.py
Standard: tests/.../test_scaled_dot_product_attention_sprint.py
```
**Always restrict to one config before debugging:**
```python
Q_CHUNK_SIZES = [288]
K_CHUNK_SIZES = [512]
```

### Running tests
```bash
tt-smi -r 0,1,2,3
pytest "...test_ring_joint_attention_sdpa_accuracy[wan2_2_compat_8544x4_h10-k512-q288-bf16]" -s
```

### If a test hangs (>15s warm cache)
```bash
# DO NOT KILL THE TEST. Run triage while it's stuck:
python3 tools/tt-triage.py > /tmp/triage.txt 2>&1
grep -E "Kernel Name|#0 |#1 |#2 " /tmp/triage.txt | head -30

# THEN kill and reset:
pkill -9 -f pytest
sleep 5
tt-smi -r 0,1,2,3
```

### Running batch PCC (one by one)
```bash
TESTS=("wan2_2_compat_8544x4_h10-k512-q288-bf16" "wan2_2_compat_2240x4_h10-k512-q224-bf16")
for t in "${TESTS[@]}"; do
    tt-smi -r 0,1,2,3 > /dev/null 2>&1
    result=$(timeout 15 pytest "...test_ring_joint_attention_sdpa_accuracy[$t]" -s 2>&1 | grep -E "PASSED|FAILED" | tail -1)
    if [ -z "$result" ]; then echo "HUNG: $t"; else echo "$result — $t"; fi
done
```
