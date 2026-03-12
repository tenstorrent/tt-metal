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

**2. Cross-ring-iteration prefetch for Q[0].**

After Q[N-1]'s save drain at the end of ring_iter R, we issue prefetch reads for Q[0] of ring_iter R+1. The reads fly during the `write_barrier` + ring device sync + mask setup. This eliminates Q[0]'s restore stall on subsequent ring iterations.

This required two insights:
- **No RAW hazard**: Q[0]'s save was written early in ring_iter R. With N-1 full Q chunk cycles elapsed (each ~500μs), the data has long landed in DRAM. A `noc_async_write_barrier()` before issuing the reads provides formal safety (completes instantly since writes departed long ago).
- **No ring_id dependency**: `get_q_chunk_info` uses `ring_id` only for `end_seq_tile`, which controls `maybe_read_tile`'s padding logic. For accumulator restore, all tiles are valid (we wrote them), so we pass `restore_end_seq_tile = 0xFFFFFFFF` to bypass padding. This decouples the prefetch from the ring sync.

**3. `writes_flushed` instead of `write_barrier` in save drain.**

The original `write_block` for output used `noc_async_write_barrier()` internally. Since save Q[i] and restore Q[i+1] hit different DRAM addresses (no RAW hazard within a ring iteration), `writes_flushed` suffices for L1 safety. The single `write_barrier` at the end of each ring iteration ensures all saves land before the next iteration reads them back.

**4. Consolidated read barrier.**

The original `read_prev_accumulators` had three separate `noc_async_read_barrier()` calls (one per CB). The new `issue_restore_reads` issues all reads across all 3 CBs, and `complete_restore` does a single barrier covering all of them.

### Implemented Writer Flow

```
// ── Ring iteration R (not last) ──

// Cross-ring prefetch from previous iteration already in flight for Q[0]
// (issued at end of ring_iter R-1, or before the loop for ring_iter 1)

for q = 0 to N-1:
    complete_restore(Q[q])               // barrier + push (instant — reads landed during prior K-loop)
    issue_restore_reads(Q[q+1])          // cb_reserve_back blocks until compute pops Q[q]'s restore
                                         // then issues reads — fly during Q[q]'s entire K-loop

    // ═══ Compute runs K-loop for Q[q] ═══
    // Prefetch reads for Q[q+1] flying in parallel

    wait_for_save_cbs(Q[q])              // blocks until compute pushes save data
    drain_save_cbs(Q[q])                 // issue writes, writes_flushed, pop

// Cross-ring prefetch: issue Q[0] reads for ring_iter R+1
noc_async_write_barrier()                // all saves landed in DRAM
issue_restore_reads(Q[0])               // reads fly during ring sync + mask setup

// ── Last ring iteration ──
// Same pattern but save is replaced by write_block for final normalized output.
// No cross-ring prefetch after the last iteration.
```

### Timeline After Prefetch

```
Writer:  [complete Q0][prefetch Q1][blocked] [save Q0][complete Q1][prefetch Q2][blocked] [save Q1] ...
Compute: [copy→K-loop Q0]                   [→save]  [copy→K-loop Q1]                   [→save]
                        ↑                                            ↑
                  reads issued here                            reads issued here
                  fly during K-loop Q0                         fly during K-loop Q1
                  barrier instant at complete Q1               barrier instant at complete Q2
```

All Q chunks (including Q[0] via cross-ring prefetch) have their restore data prefetched during the prior Q chunk's K-loop. The `complete_restore` barrier should be instant.

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

```bash
./build_metal.sh && rm -rf built/tt-metal-cache*

# PCC (CCLs restored):
pytest tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy -x

# Perf table (CCLs disabled):
pytest tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table -x -s
```

### A/B Comparison

1. Stash writer changes: `git stash -- ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
2. `./build_metal.sh && rm -rf built/tt-metal-cache*`
3. Run perf test → record baseline
4. `git stash pop`
5. `./build_metal.sh && rm -rf built/tt-metal-cache*`
6. Run perf test → record prefetch
7. Compare

## PCC Results

**Configs tested** (4 devices, Blackhole, non-Galaxy 11×10 grid, 100 SDPA cores):

| Config | q_per_core | PCC | RMSE | Status |
|---|---|---|---|---|
| s=2240/q224/k512 | 1-2 | 0.9998 | 0.004756 | PASS (identical to baseline) |
| s=2240/q288/k512 | 1-2 | 0.9998 | 0.004755 | PASS (identical to baseline) |
| s=8544/q224/k512 | 4-5 | 0.9997 | 0.006372 | PASS (identical to baseline) |
| s=8544/q288/k512 | 3 | 0.9997 | 0.006372 | PASS (identical to baseline) |

All values are **bit-identical** to baseline (no prefetch), confirming the prefetch is a pure scheduling change with no effect on computed values. These results are from the original branch; re-validation on the v2 branch (after rebase) is recommended.

## Performance Results (Blackhole, 4 devices, CCLs disabled)

**s=8544/q288/k512** (the multi-Q case — q_per_core=3, all cores hit prefetch path, 6 restores per core):

| Variant | Duration | Math Util |
|---|---|---|
| Baseline | 8.375 ms | 64.6% |
| Prefetch | 8.403 ms | 64.4% |
| Delta | +0.028 ms | -0.2pp |

**s=2240/q224/k512** (most cores have q_per_core=1, prefetch only on 10/100 cores):

| Variant | Duration | Math Util |
|---|---|---|
| Baseline | 0.678 ms | 54.8% |
| Prefetch | 0.680 ms | 54.6% |
| Delta | +0.002 ms | -0.2pp |

**No measurable improvement.** Within run-to-run noise.

### Why No Impact

Per-Q-chunk restore transfers ~54 tiles (~108KB for q288) from DRAM. At ~10+ GB/s DRAM bandwidth, this takes ~10-15μs. One Q chunk's K-loop against 17 K chunks takes ~510μs (based on tracy measurements of ~150μs per 5 K chunks). The restore stall is ~2-3% of per-Q-chunk time.

Total restore overhead: 9 restores × ~15μs = ~135μs out of 8,400μs = **~1.6% of total runtime**. This is within measurement noise of the tracy-based profiling, which runs the op in a subprocess.

The prefetch correctly eliminates the stall (verified by PCC — identical results mean the same data flows through the same compute), but the stall was never large enough to measure on these workloads.

### When Prefetch Would Matter

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

### Stage 4: Transaction ID (trid) Based Barriers

Assign separate trids to save writes vs prefetch reads for fully independent barrier tracking.

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
