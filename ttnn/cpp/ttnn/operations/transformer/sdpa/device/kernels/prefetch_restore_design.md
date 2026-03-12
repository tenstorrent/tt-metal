# Prefetch Restore: Overlapping DRAM Transfers with Compute in Multi-Q-Chunk Ring SDPA

## Problem Statement

In ring SDPA with **deferred normalization** and **multiple Q chunks per core** (`q_per_core > 1`), the raw softmax accumulators (output O, row-max m, row-sum l) must round-trip through DRAM between ring iterations. Each Q chunk's accumulators are saved to DRAM after processing and restored before the next ring iteration.

Without prefetch, the restore (DRAM → L1) for each Q chunk blocks compute until the data arrives. The DRAM read latency appears as a stall on the critical path of every Q chunk in every ring iteration > 0.

```
BEFORE PREFETCH (ring_iter > 0, Q chunks 0..N-1)

Writer:  [restore Q0] [blocked] [save Q0] [restore Q1] [blocked] [save Q1] ...
Compute: [===STALL===] [K-loop Q0] [copy→save] [===STALL===] [K-loop Q1] [copy→save] ...
                                                    ^
                                         full DRAM read latency
```

## Background

### Flash Attention / Online Softmax

Standard attention computes `softmax(Q@K^T / √d) @ V`, requiring the full N×N attention matrix. Flash Attention avoids this by processing K/V in **chunks** while maintaining three running accumulators per Q row:

| Accumulator | Role |
|---|---|
| **m** (row-max) | Max of all QK^T scores seen so far (numerical stability) |
| **l** (row-sum) | Sum of exp(score − m) over all K chunks (softmax denominator) |
| **O** (output) | Weighted sum: softmax_weights @ V, un-normalized |

Per K-chunk update:
1. `S_j = Q @ K_j^T × scale`
2. `m_new = max(m_old, rowmax(S_j))`
3. Correction: `α = exp(m_old − m_new)`
4. Rescale: `l = l × α + rowsum(exp(S_j − m_new))`, `O = O × α + exp(S_j − m_new) @ V_j`

Final normalization (after all K chunks): `O_final = O / l`

### Single Q-Chunk vs Multi Q-Chunk

When each core processes exactly **one Q chunk** (`q_per_core == 1`), the accumulators live in L1 permanently. A `RingAccumulatorState` struct holds two `AccumulatorHalf` values (prev and cur, each containing CB indices for {sum, max, out}). These persist across ring iterations with zero DRAM traffic — the ping-pong CB pairs (A/B variants of out_im, max, sum) simply stay resident.

When a core has **multiple Q chunks** (`q_per_core > 1`), there's only one set of accumulator CBs but multiple Q chunks to process. After processing Q chunk i, the accumulators must be **saved to DRAM** so the CBs can be reused for Q chunk i+1. On the next ring iteration, Q chunk i's accumulators must be **restored from DRAM** before processing resumes. This DRAM round-trip is the focus of this document.

### Ping-Pong Accumulators

`AccumulatorHalf` is a struct holding three CB indices: `{sum, max, out}`. Two halves exist — `prev` and `cur` — backed by the A/B CB pairs:

| Half | sum CB | max CB | out CB |
|---|---|---|---|
| A (initially prev) | `cb_sum_A` (c_29) | `cb_max_A` (c_27) | `cb_out_im_A` (c_25) |
| B (initially cur) | `cb_sum_B` (c_30) | `cb_max_B` (c_28) | `cb_out_im_B` (c_26) |

After each K chunk, `std::swap(q_prev, q_cur)` flips the roles: what was `cur` (just written) becomes `prev` for the next iteration, and the old `prev` (consumed) becomes `cur` (to be overwritten). After all K chunks, `q_prev` holds the final accumulated state.

### SALAD Correction

SALAD (Scale-And-correct-Last-Accumulator-Differences) is the rescaling step that adjusts old accumulators when a new K chunk produces a larger row-max. Within `sdpa_inner_loop_step`, for K chunk j > 0:

1. `sub_exp_first_col_blocks(prev.max, cur.max, cb_exp_max_diff)` — computes `α = exp(m_old − m_new)` per row (only the first column of the max tile is meaningful, since reduce_block_max_row puts the result there)
2. `mul_bcast_cols_l1_acc(prev.sum, cb_exp_max_diff, cur.sum)` — rescales old sum: `l_new += l_old × α`
3. `mul_block_bcast_cols_acc(prev.out, cb_exp_max_diff, cur.out)` — rescales old output: `O_new += O_old × α`

This correction is overlapped with V matmul on a row-by-row basis (SALAD for row i happens while V matmul for row i+1 is in progress).

### Deferred Normalization in Ring SDPA

Prior to the deferred norm change, each ring iteration performed a sigmoid-based merge to combine the current iteration's result with the prior iteration's. This required computing LSE = m + log(l), sigmoid, logsigmoid — ~65 lines of compute per ring iteration.

With deferred normalization (commit a8cf1ba5), the raw accumulators (O, m, l) carry forward across ring iterations. The SALAD correction within `sdpa_inner_loop_step` treats the first K chunk of ring iteration i+1 as a continuation from ring iteration i. Normalization (`O / l`) happens **once** on the last K chunk of the last ring iteration — `normalize_row_streaming` computes `recip(l)` then `O × recip` via column-broadcast multiply. This reduced compute code size by 40% and improved math utilization by 1–2pp.

### NOC Transfer Semantics

| Primitive | Guarantees | L1 safe? | DRAM arrived? |
|---|---|---|---|
| `noc_async_writes_flushed()` | Writes departed core (DMA read L1 → NOC) | **Yes** | No |
| `noc_async_write_barrier()` | Writes completed at destination | Yes | **Yes** |
| `noc_async_read_barrier()` | Reads completed (data in L1) | N/A | N/A — data in L1 |

Key properties:
- **Read and write queues are independent.** Reads fly without waiting for writes, and vice versa.
- **`noc_async_writes_flushed()`** guarantees the DMA engine has finished reading the source L1 data. The L1 buffer is safe to reuse. The data may still be in-transit on the NOC fabric.
- **`noc_async_write_barrier()`** guarantees writes arrived at the destination. Needed only when reading back from the **same DRAM address**.
- **Transaction IDs (trid)**: 16 available (0x0–0xF). Per-trid barriers exist for fine-grained control.

### Circular Buffer Mechanics

- `cb_pop_front` advances the read pointer, making space available for `cb_reserve_back`. A subsequent reserve + pack **can overwrite** the freed L1 region. Therefore, any in-flight DMA reading from that L1 must complete (or at least depart via `writes_flushed`) before popping.
- `cb_reserve_back` blocks until sufficient space is available (i.e., consumer has popped).
- `cb_push_back` makes tiles visible to the consumer (`cb_wait_front`).

### CB Map

| CB | Index | Direction | Contents |
|---|---|---|---|
| `cb_out` | c_16 | compute → writer | Un-normalized O (middle iters) or normalized output (last iter) |
| `cb_max_out` | c_17 | compute → writer | Row-max m |
| `cb_sum_out` | c_10 | compute → writer | Row-sum l |
| `cb_prev_out` | c_7 | writer → compute | Previous O restored from DRAM |
| `cb_max_in` | c_6 | writer → compute | Previous m restored from DRAM |
| `cb_sum_in` | c_11 | writer → compute | Previous l restored from DRAM |

Save CBs (c_16, c_17, c_10) and restore CBs (c_7, c_6, c_11) are **separate** — no aliasing.

### When Each Accumulator Is First Consumed

| Accumulator | First read | Phase | Timing |
|---|---|---|---|
| **prev.max** | `reduce_c_row_group(... prev.max ...)` | Phase 1, first q_subblock | **Immediately** — needed for eltwise max with new row-max |
| **prev.sum** | `mul_bcast_cols_l1_acc(prev.sum ...)` in SALAD | Phase 2, q_subblock 1 | After all of Phase 1 + first V matmul row |
| **prev.out** | `mul_block_bcast_cols_acc(prev.out ...)` in SALAD | Phase 2, q_subblock 1 | Same timing as prev.sum |

This gap is relevant for Stage 2 (staggered restore): **prev.max must be ready before the K-loop starts**, but **prev.out and prev.sum aren't needed until well into Phase 2**.

## Stage 1 Implementation (DONE)

### What Changed

**File**: `dataflow/ring_joint_writer.cpp` (writer-only change, compute untouched)

Split `read_prev_accumulators` into two functions:

```cpp
// Issue restore reads — non-blocking. Reserves CB space, issues NOC reads.
void issue_restore_reads(cat_out_generator, stats_writer, stats_tile_logical,
    nb, nq, Sq_chunk_t, out_slice, end_seq_tile,
    stats_seq_start_tile, stats_seq_end_tile, sum_offset,
    cb_prev_out, cb_max_in, cb_sum_in, tile_bytes, stats_tile_bytes);

// Complete a previously issued restore — single barrier + push all 3 CBs.
void complete_restore(cb_prev_out, out_num_tiles, cb_max_in, cb_sum_in, Sq_chunk_t);
```

Split `write_accumulators` into two functions:

```cpp
// Wait for compute to push all 3 save CBs. Returns when save data is ready.
void wait_for_save_cbs(Sq_chunk_t, out_slice, cb_out, cb_max_out, cb_sum_out);

// Drain save CBs to DRAM. Uses writes_flushed (not write_barrier) for L1 safety.
void drain_save_cbs(cat_out_generator, stats_writer, stats_tile_logical,
    nb, nq, Sq_chunk_t, out_slice, end_seq_tile,
    stats_seq_start_tile, stats_seq_end_tile, sum_offset,
    cb_out, cb_max_out, cb_sum_out, tile_bytes, stats_tile_bytes);
```

### Key Design Decisions

**1. Prefetch issued right after `complete_restore`, not after save drain.**

The `cb_reserve_back` inside `issue_restore_reads` naturally blocks until compute pops Q[q]'s restore data via `copy_block`. This happens early in Q[q]'s K-loop. So the reads fly during **the entire K-loop**, not just the save drain. This was an important correction during implementation — the initial approach issued prefetch after the save drain, giving only ~few μs of overlap (save write time) instead of ~500μs (full K-loop).

**2. Cross-ring-iteration prefetch for Q[0] — issued inside the Q loop at Q[N-1].**

At Q[N-1] position (last Q chunk of a ring iteration), we issue prefetch reads for Q[0] of the *next* ring iteration. This fires on **any** ring_iter (including 0), giving the reads Q[N-1]'s entire K-loop (~700us) to fly — the same overlap as intra-ring prefetches.

Key insights:
- **No RAW hazard**: Q[0]'s save was written early in the current ring iteration. A `noc_async_write_barrier()` before issuing the reads ensures DRAM arrival (completes instantly since writes departed hundreds of us ago).
- **No ring_id dependency**: `get_q_chunk_info` uses `ring_id` only for `end_seq_tile`, which controls `maybe_read_tile`'s padding logic. For accumulator restore, all tiles are valid (we wrote them), so we pass `restore_end_seq_tile = 0xFFFFFFFF` to bypass padding. This decouples the prefetch from the ring sync.
- **Ring_iter 0 safety**: On ring_iter 0, the restore CBs have never been used, so `cb_reserve_back` succeeds immediately. Compute never calls `copy_block` for restore on ring_iter 0, so the data sits in the CBs until ring_iter 1's `complete_restore` pushes it.
- **Intra-ring prefetch must NOT fire on ring_iter 0**: Q[q+1]'s accumulators don't exist in DRAM yet (they haven't been saved). Only the cross-ring prefetch (Q[0]'s save from the current iteration) is valid on ring_iter 0. An earlier attempt that also issued intra-ring prefetches on ring_iter 0 read garbage from DRAM, causing PCC failure (RMSE 0.287).

**3. `writes_flushed` instead of `write_barrier` in save drain.**

The original `write_block` for output used `noc_async_write_barrier()` internally. Since save Q[i] and restore Q[i+1] hit different DRAM addresses (no RAW hazard within a ring iteration), `writes_flushed` suffices for L1 safety. The single `write_barrier` at the end of each ring iteration ensures all saves land before the next iteration reads them back.

**4. Consolidated read barrier.**

The original `read_prev_accumulators` had three separate `noc_async_read_barrier()` calls (one per CB). The new `issue_restore_reads` issues all reads across all 3 CBs, and `complete_restore` does a single barrier covering all of them.

### Implemented Writer Flow

```
// ── Ring iteration 0 ──

for q = 0 to N-1:
    // No complete_restore (ring_iter == 0 — nothing to restore)
    // No intra-ring prefetch (Q[q+1]'s accumulators don't exist in DRAM yet)

    if q == N-1 and not last_ring_iter:
        write_barrier()                  // ensure Q[0]'s save landed (instant — saved long ago)
        issue_restore_reads(Q[0])        // cross-ring: reads fly during Q[N-1]'s K-loop

    // ═══ Compute runs K-loop for Q[q] ═══
    wait_for_save_cbs(Q[q])
    drain_save_cbs(Q[q])

write_barrier()

// ── Ring iteration R > 0 (not last) ──

for q = 0 to N-1:
    complete_restore(Q[q])               // barrier + push (instant — reads landed during prior K-loop)
    issue_restore_reads(Q[q+1])          // intra-ring: blocks until compute pops Q[q]'s restore,
                                         // then reads fly during Q[q]'s entire K-loop

    if q == N-1 and not last_ring_iter:
        write_barrier()                  // ensure Q[0]'s save landed (instant)
        issue_restore_reads(Q[0])        // cross-ring: reads fly during Q[N-1]'s K-loop

    // ═══ Compute runs K-loop for Q[q] ═══
    wait_for_save_cbs(Q[q])
    drain_save_cbs(Q[q])

write_barrier()

// ── Last ring iteration ──
// Same as R > 0 but save is replaced by write_block for final normalized output.
// No cross-ring prefetch (no next iteration).
```

### Detailed Timeline (q_per_core=3, Q chunks 0,1,2)

```
═══════════════════════════════════════════════════════════════════════
 RING ITER 0  (no restores — accumulators don't exist in DRAM yet)
═══════════════════════════════════════════════════════════════════════

  Q[0]:
    Writer:  (nothing)
    Compute: [────── K-loop Q[0] ──────][copy→save CBs]
    Writer:  [wait_for_save][drain_save Q[0] → DRAM]

  Q[1]:
    Writer:  (nothing)
    Compute: [────── K-loop Q[1] ──────][copy→save CBs]
    Writer:  [wait_for_save][drain_save Q[1] → DRAM]

  Q[2] (last Q chunk → cross-ring prefetch fires):
    Writer:  [write_barrier][issue_restore_reads Q[0] for next ring iter]
                                  │
                                  ▼ reads fly during Q[2]'s K-loop ──────────┐
    Compute: [────────────── K-loop Q[2] ──────────────────][copy→save CBs]  │
    Writer:  [wait_for_save][drain_save Q[2] → DRAM]                         │
                                                                              │
  write_barrier() ◄── ensures all saves landed                               │
                                                                              │
═══════════════════════════════════════════════════════════════════════       │
 RING ITER 1  (ring sync + mask setup happens here)                          │
═══════════════════════════════════════════════════════════════════════       │
                                                                              │
  Q[0]:                                                                       │
    Writer:  [complete_restore Q[0]]  ◄── barrier instant (reads flew above)──┘
             [issue_restore_reads Q[1]]
                     │
                     ▼ cb_reserve_back blocks until compute pops Q[0]'s restore
                       then reads fly during Q[0]'s K-loop ─────────────┐
    Compute: [copy_block restore→accum][─── K-loop Q[0] ───][copy→save] │
    Writer:  [wait_for_save][drain_save Q[0] → DRAM]                     │
                                                                          │
  Q[1]:                                                                   │
    Writer:  [complete_restore Q[1]]  ◄── instant ────────────────────────┘
             [issue_restore_reads Q[2]]
                     │
                     ▼ reads fly during Q[1]'s K-loop ─────────────────┐
    Compute: [copy_block restore→accum][─── K-loop Q[1] ───][copy→save]│
    Writer:  [wait_for_save][drain_save Q[1] → DRAM]                    │
                                                                         │
  Q[2] (last Q → cross-ring prefetch):                                   │
    Writer:  [complete_restore Q[2]]  ◄── instant ───────────────────────┘
             [write_barrier][issue_restore_reads Q[0] for next ring iter]
                                  │
                                  ▼ reads fly during Q[2]'s K-loop ────────┐
    Compute: [copy_block restore→accum][─── K-loop Q[2] ───][copy→save]   │
    Writer:  [wait_for_save][drain_save Q[2] → DRAM]                       │
                                                                            │
  write_barrier()                                                           │
                                                                            │
═══════════════════════════════════════════════════════════════════════     │
 LAST RING ITER                                                            │
═══════════════════════════════════════════════════════════════════════     │
                                                                            │
  Q[0]:                                                                     │
    Writer:  [complete_restore Q[0]]  ◄── instant ──────────────────────────┘
             [issue_restore_reads Q[1]]
                     │
                     ▼ reads fly during Q[0]'s K-loop ─────────────┐
    Compute: [copy_block][─── K-loop Q[0] ───][normalize → cb_out] │
    Writer:  [write_block final output → DRAM]                      │
                                                                     │
  Q[1]:                                                              │
    Writer:  [complete_restore Q[1]]  ◄── instant ───────────────────┘
             [issue_restore_reads Q[2]]
                     ▼ ...                                          ┐
    Compute: [copy_block][─── K-loop Q[1] ───][normalize → cb_out] │
    Writer:  [write_block final output → DRAM]                      │
                                                                     │
  Q[2]:                                                              │
    Writer:  [complete_restore Q[2]]  ◄── instant ───────────────────┘
             (no cross-ring prefetch — last ring iter)
    Compute: [copy_block][─── K-loop Q[2] ───][normalize → cb_out]
    Writer:  [write_block final output → DRAM]

  write_barrier()
```

Key points:
- Every Q chunk's restore is prefetched during the *previous* Q chunk's K-loop (~500-700μs of overlap), so `complete_restore`'s `noc_async_read_barrier()` is instant.
- Q[0] is special: its prefetch comes from the *cross-ring* path at Q[N-1] of the prior ring iteration, giving equally long overlap.
- Ring iter 0 has no restores (accumulators don't exist yet), only the cross-ring prefetch for Q[0] of ring iter 1.
- `issue_restore_reads` blocks on `cb_reserve_back` until compute pops the current Q's restore data — this is the natural synchronization point.

## Branch History

### Original branch: `dnijemcevic/sdpa_deferred_norm_prefetch` (5 commits)

| # | Commit | Status |
|---|---|---|
| 1 | `eddb25f712` AccumulatorHalf struct | Landed on main via PR #39540 |
| 2 | `24b9c89942` Deferred normalization | Landed on main via PR #39540 |
| 3 | `7083b6cb08` Remove noinline/noclone | Dropped (belongs to `sdpa_code_size_sweep` branch) |
| 4 | `e093a88790` Fix q_per_core mismatch | Landed on main via PR #39540 |
| 5 | `6aa80ae120` Prefetch restore | **Unique — cherry-picked to v2** |

### Current branch: `dnijemcevic/sdpa_deferred_norm_prefetch_v2` (1 commit on main)

Created by cherry-picking commit 5 onto `origin/main`. Three conflicts resolved in `ring_joint_writer.cpp`:

1. **`drain_save_cbs` max/sum write section**: Main had per-CB `writes_flushed` + immediate `cb_pop_front`; resolved to our consolidated single `writes_flushed` + batch pop (the whole point of the prefetch refactor).
2. Same pattern for the sum section.
3. **`is_last_ring_iter`**: Main uses `last_active_ring_iter` (from `find_last_active_ring_iter`), which is correct when ring iterations can be skipped (e.g., sequence length doesn't fill all ring shards). Kept main's version instead of the original `ring_size - 1`.

### Other PRs that landed on main since the original branch

| PR | Change | Impact on prefetch |
|---|---|---|
| #39540 | Deferred normalization | Base for this work — already integrated |
| #39125 | Mcast support + chain deadlock fix | Factory changes only — no conflict with writer prefetch logic |
| #39420 | Performance model | Factory changes only — no conflict |

## Testing

### Test File

`tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py`

This is the mainlined test (committed on main). It contains 4 tests, all parameterized by `INPUT_SHAPES × Q_CHUNK_SIZES × K_CHUNK_SIZES`. Perf tests (1, 4) are `@pytest.mark.skipif(CI)` — skipped on CI, run locally.

### Configuration Constants

```python
NON_GALAXY_SEQ_LENS_PER_DEVICE = [8544]   # original: [2240, 8544]
Q_CHUNK_SIZES = [288]                      # original: [224, 256, 288]
K_CHUNK_SIZES = [512]                      # original: [128, 256, 512]
```

Currently hacked to a single config: `s=8544×sp_size, q288, k512` — the multi-Q-chunk case. To sweep the full matrix, restore the original values (commented out in the file). The hack is an **unstaged local modification** — do not commit it.

Key derived values for non-Galaxy (4-device, 11×10 grid):
- `local_seq_len = 8544` (per device), `total_seq_len = 8544 × 4 = 34176`
- SDPA cores: 100 (10 columns × 10 rows; column 10 reserved for CCL)
- `q_num_chunks = ceil(8544 / 288) = 30`, parallelized across 10 heads × 10 Q-parallel = 100 cores
- `q_per_core = ceil(30 / 10) = 3` on all cores

### Test Descriptions

| Test | Purpose | Needs CCLs? | How it runs |
|---|---|---|---|
| `test_ring_joint_attention_sdpa_sweep_perf_impl` | Raw kernel run, `do_check=False` | No | Direct pytest |
| `test_ring_joint_attention_sdpa_accuracy` | PCC + RMSE vs PyTorch `F.scaled_dot_product_attention` | **Yes** | Direct pytest |
| `test_ring_joint_attention_sdpa_determinism` | 10 runs, bitwise comparison | Yes | Direct pytest |
| `test_ring_joint_attention_create_perf_table` | Tracy-profiled math util sweep | No | Spawns subprocess via `run_device_profiler()` |

### Math Utilization Formula (test 4)

```
mm_flops = 4 × local_seq_len × total_seq_len × head_dim × heads_per_device
cycles = duration_ns × 1.35 GHz
theoretical_flops = effective_cores × cycles × 2048 (FLOPs/cycle/core)
math_util = mm_flops / theoretical_flops × 100
```

The `4×` accounts for two matmuls (Q@K^T and QK^T@V), each with 2N multiplies for an N×N tile.

`duration_ns` is the **max** across all cores from tracy's `DEVICE KERNEL DURATION [ns]` column. `effective_cores` is the measured core count rounded down to the nearest 10 (to exclude CCL cores from the utilization denominator).

### PCC Measurement (test 2)

Runs the op directly (no subprocess). Compares output vs `torch.nn.functional.scaled_dot_product_attention` on full combined Q/K/V (simulating all-gather). Thresholds: PCC >= 0.994, RMSE < 0.05.

### CCL Toggle for Perf vs PCC

The CCL reader/writer kernels control cross-device data movement. Commenting them out makes each device process only its local KV shard + garbage from remote shards — **incorrect for PCC, but isolates compute+DRAM perf**.

**Current working tree state**: CCLs commented out (from commit `d327858d`). Two files are modified but unstaged:
```
ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp
ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_writer.cpp
```

**To restore CCLs for PCC testing**:
```bash
git checkout -- ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/
```

**To re-disable CCLs for perf testing**:
```bash
git show d327858d008c44abe1f1433d098007a785d2ef07:ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp > ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp
git show d327858d008c44abe1f1433d098007a785d2ef07:ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_writer.cpp > ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_writer.cpp
```

### Build & Run

Kernel changes (`.cpp`/`.hpp` under `kernels/`) are JIT-compiled — only need `rm -rf built/tt-metal-cache*`, not a full `./build_metal.sh`. Full rebuild is needed for host-side changes (factory, Python bindings).

```bash
# After kernel-only changes:
rm -rf built/tt-metal-cache*

# After host-side changes:
./build_metal.sh && rm -rf built/tt-metal-cache*

# PCC (CCLs restored):
source python_env/bin/activate
pytest tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy -x

# Perf table with math util (CCLs disabled, uses Tracy subprocess):
source python_env/bin/activate
export TT_METAL_DEVICE_PROFILER=1
pytest tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -x -s
```

### Generating Tracy Profiles for the GUI

To produce a `.tracy` file for the Tracy profiler GUI, run the perf test wrapped with `python -m tracy`:

```bash
source python_env/bin/activate
rm -rf built/tt-metal-cache*
export TT_METAL_DEVICE_PROFILER=1
python -m tracy -r -p -n <run_name> \
  -m pytest "tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl[wan2_2_compat_8544x4_h10-k512-q288-bf16]"
```

Flags:
- `-r` — generate report (starts `capture-release`, produces `.tracy` + CSV)
- `-p` — partial profiling (only enabled zones, less overhead)
- `-n <run_name>` — names the output folder

Output location:
```
generated/profiler/reports/<run_name>/<timestamp>/tracy_profile_log_host.tracy
generated/profiler/reports/<run_name>/<timestamp>/profile_log_device.csv
generated/profiler/reports/<run_name>/<timestamp>/ops_perf_results_*.csv
```

The `.tracy` file can be opened in the Tracy GUI (`tracy` or `Tracy-release`). The `profile_log_device.csv` has per-core per-zone timestamps (ZONE_START/ZONE_END with cycle counts at 1.35 GHz).

**Profiling markers**: `DeviceZoneScopedN("name")` in compute kernels appears on TRISC threads; in writer kernels it appears on **BRISC** (not NCRISC). Markers add overhead (~1-2%), so comment them out for clean perf measurement and uncomment for profiling analysis.

**Board resets**: If tests hang with "Timed out while waiting for active ethernet core", reset boards with `tt-smi -r 0,1,2,3`. This can happen intermittently with commented-out CCLs.

### A/B Comparison

1. Stash writer changes: `git stash -- ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
2. `rm -rf built/tt-metal-cache*`
3. Run perf table test → record baseline
4. `git stash pop`
5. `rm -rf built/tt-metal-cache*`
6. Run perf table test → record prefetch
7. Compare

For Tracy profile comparison, use `python -m tracy -r -p -n before ...` / `-n after ...` and compare the two `.tracy` files in the GUI.

## PCC Results

**Configs tested** (4 devices, Blackhole, non-Galaxy 11×10 grid, 100 SDPA cores):

| Config | q_per_core | PCC | RMSE | Status |
|---|---|---|---|---|
| s=2240/q224/k512 | 1-2 | 0.9998 | 0.004756 | PASS (identical to baseline) |
| s=2240/q288/k512 | 1-2 | 0.9998 | 0.004755 | PASS (identical to baseline) |
| s=8544/q224/k512 | 4-5 | 0.9997 | 0.006372 | PASS (identical to baseline) |
| s=8544/q288/k512 | 3 | 0.9997 | 0.006372 | PASS (identical to baseline) |

All values are **bit-identical** to baseline (no prefetch), confirming the prefetch is a pure scheduling change with no effect on computed values.

**v2 branch re-validation** (s=8544/q288/k512 with in-loop cross-ring prefetch):
- PCC: 0.9997, RMSE: 0.006372 — **PASS** (identical to baseline)

## Performance Results (Blackhole, 4 devices, CCLs disabled)

Measured via `test_ring_joint_attention_create_perf_table` (Tracy subprocess profiling, no profiling markers). All DeviceZoneScopedN markers were commented out for clean measurement.

**s=8544/q288/k512** (q_per_core=3, 100 SDPA cores, 17 K chunks × 4 ring iters):

| Variant | Duration | Math Util |
|---|---|---|
| No prefetch (main baseline) | 8.574 ms | 63.1% |
| In-loop cross-ring prefetch (v2) | 8.508 ms | 63.6% |
| Delta | **-0.066 ms (-0.8%)** | **+0.5pp** |

### Tracy-Measured K CHUNK Gaps (TRISC_0, core (1,2), device 0)

Ring iteration boundaries show the stall between Q[N-1]'s last K chunk and Q[0]'s first K chunk:

| Boundary | No Prefetch (us) | V2 Prefetch (us) | Saved (us) |
|---|---|---|---|
| Ring 0→1 | 12.7 | 9.0 | 3.7 |
| Ring 1→2 | 12.1 | 8.9 | 3.2 |
| Ring 2→3 | 12.1 | 8.1 | 4.0 |
| Ring 3→last | 11.5 | 7.9 | 3.6 |
| **Total** | **48.4** | **33.9** | **~14.5** |

Intra-ring Q[q]→Q[q+1] gaps (~2.4us) are unchanged — these are the irreducible `copy_block` overhead.

### Why Small Impact

The cross-ring prefetch saves ~3-4us per ring boundary by eliminating the DRAM read latency from `complete_restore`. The remaining ~8-9us gap is structural overhead: Q chunk teardown (pop Q, swap accumulators) + `copy_block` transferring restored data from CBs to accumulator CBs through DST.

Per-Q-chunk restore transfers ~54 tiles (~108KB for q288) from DRAM. At ~10+ GB/s DRAM bandwidth, this takes ~10-15μs. The prefetch hides this latency but can't eliminate the `copy_block` time.

### Tracy Profiles

Profiles with RING ITER, Q CHUNK, K CHUNK, and WR-* markers (these add ~1-2% overhead):

| Profile | File |
|---|---|
| No prefetch (baseline) | `sdpa_s8544_q288_k512_no_prefetch.tracy` (repo root) |
| V2 in-loop cross-ring prefetch | `sdpa_s8544_q288_k512_prefetch.tracy` (repo root) |

### When Prefetch Would Matter More

The restore overhead grows as a fraction of total time when:
- **Fewer K chunks per ring iteration** (shorter sequences, larger k_chunk) — K-loop is shorter, restore fraction is larger
- **More Q chunks per core** (more heads, fewer cores) — more restores per ring iteration
- **Larger accumulator sizes** (bigger Sq_chunk_t or DHt) — more data per restore

## Future Optimizations

### Stage 2: Staggered Restore

Since prev.max is needed in Phase 1 (QKT matmul) while prev.out and prev.sum aren't needed until Phase 2 (SALAD correction), the restore could be split:
- Push `cb_max_in` first → compute starts Phase 1
- `cb_prev_out` and `cb_sum_in` arrive during Phase 1

This would provide latency hiding even when the K-loop is short, at the cost of more complex synchronization (separate barriers for max vs out/sum).

### Stage 3: Row-Granular Streaming

Currently all transfers are chunk-granular (full Sq_chunk_t at once). Restructuring to work at subblock_h granularity would enable finer-grained overlap.

### Stage 4: Transaction ID (trid) Based Write Barriers

#### Problem: Blanket Write Barriers Wait for Unrelated Saves

The current code has two blanket `noc_async_write_barrier()` calls per ring iteration:

1. **Cross-ring prefetch barrier** (line 517): Fires at Q[N-1] before issuing cross-ring prefetch reads for Q[0]. Only *needs* Q[0]'s save to have arrived at DRAM, but waits for ALL pending writes — including Q[N-2]'s save, which departed L1 just ~100ns ago via `writes_flushed` in `drain_save_cbs`.

2. **End-of-ring barrier** (line 576): Fires after all Q chunks. Only *needs* Q[N-1]'s save to have arrived, but waits for everything.

The cross-ring barrier (1) is potentially the worse offender: Q[N-2]'s save (~108KB of accumulator data) has had almost no time to traverse the NOC and arrive at DRAM. Q[0]'s save (the one actually needed) landed hundreds of μs ago during Q[1]'s K-loop.

#### Measurement: End-of-Ring Barrier (line 576)

Measured with `DeviceZoneScopedN("WR-END-BARRIER")` on s=8544/q288/k512 (q_per_core=3, 4 devices, 1600 samples across all cores):

| Stat | Duration |
|---|---|
| p0 (min) | 0.03 μs |
| p25 | 0.10 μs |
| p50 | 0.26 μs |
| p75 | 0.50 μs |
| p90 | 0.88 μs |
| p99 | 2.50 μs |
| p100 (max) | 7.94 μs |

The barrier is nearly free in this config — Q[N-1]'s save has had Q[N-1]'s entire K-loop to land. However, with only 3 barriers per core (4 ring iters, last takes a different path), statistics are thin. The 7.94μs max could represent a real stall under NOC congestion.

**TODO**: Measure the cross-ring prefetch barrier (line 517) similarly. This one is more likely to show real stalls since Q[N-2]'s save is so recent.

#### Proposed Solution: Per-Save Transaction IDs

The NOC API provides per-trid write barriers:
- `noc_async_write_set_trid(trid)` — tag subsequent writes
- `noc_async_write_barrier_with_trid(trid)` — block until trid's writes arrive at DRAM
- `noc_async_write_flushed_with_trid(trid)` — block until trid's writes depart L1

**Two-trid scheme** (TRID_Q0=0, TRID_REST=1):

```
drain_save_cbs(Q[q]):
    noc_async_write_set_trid(q == 0 ? TRID_Q0 : TRID_REST);
    // ... issue writes ...
    noc_async_write_flushed_with_trid(...);   // L1 safe

Cross-ring prefetch at Q[N-1]:
    noc_async_write_barrier_with_trid(TRID_Q0);   // only Q[0]'s save — instant
    issue_restore_reads(Q[0]);

End of ring:
    (no barrier — drain_save_cbs already did writes_flushed)

Next ring iter, Q[0]:
    complete_restore(Q[0]);
    noc_async_write_barrier_with_trid(TRID_REST);  // Q[1]..Q[N-1] saves — instant
    issue_restore_reads(Q[1]);
```

#### Rejected: Move Barrier Without Trids

Moving the end-of-ring blanket `write_barrier()` to right before the first intra-ring `issue_restore_reads` in the next ring iteration would give Q[N-1]'s save more time to land. However, this only helps the end-of-ring barrier — it cannot fix the cross-ring barrier (line 517), which is the more problematic one (Q[N-2]'s save departed ~100ns ago). Trids are required to solve both.

#### Implementation Notes

- `noc_async_write_set_trid` is stateful — once set, all subsequent writes use that trid until changed. Must restore default trid (0) or set explicitly before each drain.
- The cross-ring prefetch's existing `noc_async_write_barrier()` should change to `noc_async_write_barrier_with_trid(TRID_Q0)`. This is the higher-value change (avoids waiting for Q[N-2]'s recent save).
- `write_block` (used for final output on last ring iter) uses its own internal `noc_async_write_barrier()`. This is fine — it's not in the prefetch path.
- Ring iter 0 has no restores, so trid barriers are skipped (same guards as current prefetch logic).
- PCC must be re-validated after changes — trid misuse could cause RAW hazards.

## Key Files

All paths relative to `ttnn/cpp/ttnn/operations/transformer/sdpa/device/`.

| File | Role |
|---|---|
| `kernels/compute/compute_streaming.hpp` | `sdpa_ring_v2` — compute outer loop (restore copy_blocks, K-loop, save copy_blocks). `sdpa_inner_loop_step` — one K-chunk of flash attention (Phase 1: Q@K^T + softmax, Phase 2: QK^T@V + SALAD corrections). |
| `kernels/compute/compute_common.hpp` | `copy_block`, `sub_exp_block_bcast_cols_inplace`, `reduce_c`, `mul_block_bcast_cols`, `LightweightMaskContext` |
| `kernels/compute/ring_joint_sdpa.cpp` | Compute kernel entry point — ring iteration loop, dispatches to `sdpa_ring_v2` |
| `kernels/dataflow/ring_joint_writer.cpp` | Writer kernel — DRAM read/write orchestration, **modified by prefetch** |
| `kernels/dataflow/ring_joint_reader.cpp` | Reader kernel — Q/K/V tile DMA from DRAM to CBs |
| `kernels/dataflow/dataflow_common.hpp` | `read_block`, `write_block` — bulk DRAM transfer primitives |
| `kernels/dataflow/ring_utils.hpp` | `RingIdSequencer`, `find_last_active_ring_iter` |
| `ring_joint_sdpa_program_factory.cpp` | Host-side CB allocation, compile-time args, chain/mcast construction |

### Test file (mainlined)

`tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py` (constants hacked locally for single-config runs — unstaged)

### CCL kernels (toggled for perf vs PCC)

```
ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_reader.cpp
ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_all_gather_writer.cpp
```

## Sources

- Flash Attention: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (arXiv:2205.14135)
- DeepWiki research on `tenstorrent/tt-metal`: NOC barrier semantics, `writes_flushed` vs `write_barrier`, transaction IDs, CB mechanics
- PR #39540: "Add deferred normalization to ring SDPA" (mainlined version of commits 1-2, 4)
- Commit d327858d: "Comment out CCLs in ring joint SDPA" (perf testing baseline)
