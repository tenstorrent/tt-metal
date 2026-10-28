# Changelog: clamp_square_practice

## Phase 0 — Core Implementation
- **Date**: 2026-10-28
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier). It is
  one `ttnn.generic_op` dispatch:
  - The flattened tile axis is split across the full grid (`split_work_to_cores(row_wise=True)`).
  - The reader and writer are block-batched (one barrier or flush per block).
  - Compute is one fused `eltwise_chain`: Copy → Clamp(fp32 RT bounds) → Square → Pack.
- **SUPPORTED at Phase 0**: dtype=[bfloat16], layout=[TILE], memory_layout=[INTERLEAVED], buffer_type=[DRAM],
  rank=[rank_ge_2], alignment=[tile_aligned]
- **Accuracy achieved**:
  - bf16-exact bounds: PCC=0.99999943, max_abs_err=1.94e-3, rel_rms_err=4.85e-4, **0 ULP** (bit-exact vs the
    bf16-rounded fp32 reference).
  - Non-bf16-exact bounds (-0.3, 0.7): PCC=0.9999983, max_abs_err=1.72e-3, rel_rms=3.6e-3, ≤ 1 ULP.
  - Measured on 4 shapes × 3 bound sets via `test_clamp_square_practice_precision_baseline.py`.
- **Golden suite at Phase 0**: n/a. `eval/golden_tests/clamp_square_practice/` does not exist, so `verify_supported`
  was not run. Acceptance is 48/48 on Blackhole.
- **Perf baseline (BH, 120 cores)**:
  - `(2,4,256,512)`: 13.75 µs (305 GB/s)
  - `(8,1,1024,1024)`: 87.2 µs (385 GB/s)
  - `(1,1,4096,4096)`: 182.5 µs (368 GB/s)
- **Issues encountered**: verifier fix — finite `min_val`/`max_val` beyond the fp32 range escaped as `OverflowError`
  from the host fp32 packer. `validate()` now rejects them with `ValueError`, consistent with `torch.clamp`.
- **Tests added**: `test_clamp_square_practice.py` (planner), `test_clamp_square_practice_precision_baseline.py`,
  `test_clamp_square_practice_extended.py` (verifier).

## Perf 1 — perf tournament round 1
- **Date**: 2026-10-28. Blackhole p150b, 120 cores, 1350 MHz. Metric: DEVICE KERNEL DURATION (one fresh-cache
  profiled run per variant). SUPPORTED is unchanged.
- **Focus shape**: `feature_spec.py` has no `LOOSE_CASES` perf flag, so I picked the largest golden shape,
  `(1,1,4096,4096)` (16 384 tiles, ~137 tiles/core). I also measured `(8,1,1024,1024)`, `(2,4,256,512)` and
  `(1,1,32,32)`.
- **Instrumentation (permanent)**: `MaybeDeviceZoneScope` zones `reader_read`, `compute_chain`, `writer_write` and
  `writer_final_barrier`. They are kernel-level only, never per block, because of the per-RISC marker budget.
  `ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` was missing from this checkout, so I restored it verbatim
  from 9518ba45263.

### Measured breakdown
Cumulative ablation on the focus shape. Each variant stubs the payload and keeps all sync and trip counts. Bench:
`tests/ttnn/unit_tests/operations/clamp_square_practice/test_perf1_breakdown.py`.

| variant | ns |
|---|---|
| full | 185 534 |
| read + write (compute stubbed) | 159 524 |
| read + compute | 113 840 |
| write + compute | 151 698 |
| read only | 83 798 |
| write only | 111 543 |
| compute only | 79 150 (~780 cycles/tile) |
| sync floor | 1 817 |

- **Ranked bottleneck**:
  1. DRAM-bound DM. 64 MiB moves in 159.5 µs, about 420 GB/s. The ceiling (`noc_balance` agent, from
     `noc_latencies.yaml`) is a 131 µs DRAM floor at the 512 GB/s peak, or 142.5 µs at a practical 92%.
     DM-only is therefore at 1.10× its target.
  2. Compute is **not hidden**: it adds +26 µs over DM-only. Compute alone is 79 µs: Clamp 453 and Square
     258 cycles/tile, with copy+pack+sync at only ~44 cycles/tile.
  3. Strong spatial per-core skew at equal work. Writer span is p50 125 µs and max 180 µs; the east columns and
     top rows are slowest.

### Portfolio and verdicts
Focus-shape whole-op ns. Each experiment's directory is `perf_experiments/<slug>/`.

| idea | verdict | before → after | domain / notes |
|---|---|---|---|
| `fused_sfpu_clamp_square`: one raw SFPU pass (load, 2× SFPSWAP vs const regs, SFPMUL, 4-op RNE, store) | **WIN** | compute-isolated 78.6 → 40.5 µs (1.94×); whole op 181–186 → 164.5 µs | All shapes measured. Bit-identical to the helper chain for bf16-exact bounds (exhaustive 65 536 bf16 inputs). Inexpressible under fp32 DEST (`static_assert`); the op hard-codes 16-bit DEST. Options not graduated: SFPLOADMACRO schedule, 37.6 µs isolated, ties on the whole op and leaves SFPU macro state programmed; SFPSTOCHRND rounding, 30.7 µs, **breaks precision** (1 ULP on 1–3% of elements). |
| `compute_datapath_overhead`: unpack-to-DEST copy, block copy/pack, init hoist, one SFPU bracket per block | **NULL** (≤ 2% isolated, noise on whole op) | — | Unpack-to-DEST is not available for 16-bit DEST. Side finding: hoisting Clamp's bounds into const regs gives 1.50× isolated and 1.07× whole op, bit-identical. **Superseded** by the fused kernel, which already includes it. |
| `pipeline_depth`: CB depth / DM block decoupled from DEST block / coarser flush | **WIN** | 184.3 → 159.4 µs with depth 32 blocks at block 8 | Wins from ~69 tiles/core up; flat (noise) below ~18 tiles/core; no measured regression. Decoupling the DM block (16/32/64) and barrier-every-k were 3–10% worse than block 8. Correct on ragged tails. |
| `noc_balance`: per-core writer NoC by grid diagonal, read NoC by hop count, alternating NoCs, bank stagger, strided pages | not graduated | whole op 183.3 → ~171 µs median (±5%; 1 run in 10 at baseline); DM-only 157 → 153.7 µs | Win is noisy, and the diagonal threshold was fitted to this board's harvested grid, i.e. a predicate over what was measured. Once compute is hidden the op sits at the DM floor, where this moves only ~2%. Recorded for round 2. Alternate-per-page and bank stagger: flat. Strided, hop-count writes and whole-stream flips: regressions. |
| `work_rebalance`: dynamic block claiming via NoC fetch-and-add; strided ownership; fewer cores | **BLOCKED** (dynamic) / NULL (rest) | dynamic: 184.0 → 147.8 µs, all cores within 1 µs | **Incorrect**: about 1 in 200 launches a claim returns a duplicate value and a block is never written (evidence in `work_rebalance/evidence_claimlog.log`). There was also one hang. Root cause unknown; it needs a verified fetch-and-add on BH. Strided and skew: flat. 64/80/100 static cores: slower. |

### Graduated
1. **Fused raw-SFPU clamp² chain element** (`ClampSquareFused`, plain raw-TTI schedule) inside the existing
   `eltwise_chain` (CopyTile → ClampSquareFused → PackTile).
   - It is the op's only compute path. The helper `Clamp` → `Square` elements were deleted.
   - No carve-out: fp32 DEST is unreachable, since the op hard-codes `fp32_dest_acc_en=False`, and a
     `static_assert` guards it.
   - Precision: bit-identical for bf16-exact bounds. For non-bf16-exact bounds, the old path squared the bound
     after it was truncated to bf16 in DEST, which put it 1 ULP off the op's fp32 reference on clamped
     elements. The new path squares the fp32 bound and matches `torch.clamp(x.float(), lo, hi)**2 → bf16`
     exactly, so it is more accurate.
   - Golden and precision-baseline tests pass.
2. **CB depth 2 → 32 blocks**, capped at the per-core assignment rounded up to a whole block. This is
   descriptor-only.
   - It is the only path, with no carve-out. The cap is not a guard: a CB that holds the whole assignment can
     never block, so small shapes simply don't allocate what they can't use.
   - L1 cost: at most 1 MiB per core; 576 KiB at the focus shape.
3. All four zones are carried to the new path, with `compute_chain` on the fused kernel.

### Whole-op before → after and guard set
Baseline = HEAD compute kernel + 2-block CBs; same bench, same session. All shapes pass the golden suite.

| shape | before ns | after ns | Δ |
|---|---|---|---|
| (1,1,4096,4096) **focus** | 181 424 | 157 833 | −13.0% |
| (8,1,1024,1024) | 90 027 | 81 341 | −9.6% |
| (2,4,256,512) | 14 244 | 12 511 | −12.2% |
| (1,1,256,256) | 2 933 | 2 654 | −9.5% |
| (1,1,32,8192) | 4 633 | 4 164 | −10.1% |
| (2,64,256) | 2 605 | 2 310 | −11.3% |
| (64,128) | 2 087 | 1 743 | −16.5% |
| (1,1,32,32) | 2 024 | 1 727 | −14.7% |

- **No regression on any guard cell.** The op has a single kernel path × TILE × DRAM-interleaved, so these
  shapes span its per-core-count regimes: 1 tile/core, ragged, and 137 tiles/core.
- After Perf 1 the focus shape sits on its DM-only floor (157–159 µs). The writer span is p50 118 µs, max
  157 µs, so the critical path is the spatial DM skew on the slowest cores. That is the round-2 target:
  `noc_balance`, plus `work_rebalance` if a correct claim primitive exists.
- Golden `eval/golden_tests/clamp_square_practice/`: 46/46. Op unit + extended + precision-baseline tests: 68/68.

### Helper bypasses
| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `compute_kernel_lib::Clamp` + `compute_kernel_lib::Square` (eltwise chain elements → `calculate_clamp` / `calculate_square`) | capability | (1) No chain element computes a fused unary sequence (clamp then square) in **one** DEST row loop: each element is its own SFPU pass with its own load/store per row, and two distinct SFPU elements also make the chain re-run both inits every block. (2) `calculate_clamp` (BH) keeps neither bound resident: it rewrites LREG12 per bound per row (2× SFPLOADI + SFPCONFIG) instead of holding lo/hi in two programmable const regs. (3) `calculate_square`'s bf16 RNE is a 5-op software round. The fix the helpers lack is a way to compose SFPU element bodies into one row loop, or a `ClampSquare`-style fused element, plus a const-reg-resident clamp. | 78 600 (compute-isolated, focus) | 40 500 | `kernels/clamp_square_practice_compute.cpp:105` (`ClampSquareFused`; raw body `raw_face` at :78) |

The `eltwise_chain` walk, CopyTile, PackTile and the block sync are still the helper's; only the SFPU element is
custom. The kernel-head justification comment is at the bypass site.

**Summary**: 5 ideas measured. 2 graduated (fused SFPU, deep CBs). 1 null that was superseded by the fused
kernel (its clamp-hoist side win is already included). 1 noisy win not graduated (board-fitted NoC split).
1 blocked on correctness (dynamic claiming). The op is 13.0% faster on the focus shape, faster on every guard
shape, with no regression.

## Perf 2 — perf tournament round 2
- **Date**: 2026-10-28. Blackhole p150b, 120 cores. Metric: DEVICE KERNEL DURATION, one fresh-cache profiled run
  per variant (the guard set was run twice). SUPPORTED is unchanged.
- **Focus shape**: `feature_spec.py` has no `LOOSE_CASES` perf flag, so the focus stays `(1,1,4096,4096)`, the
  same as Perf 1, measured at the op's only config (bf16, TILE, DRAM interleaved, 16-bit DEST).
- **Instrumentation**: the Perf 1 zones are unchanged and carried onto the new reader and writer: `reader_read`,
  `compute_chain`, `writer_write` and `writer_final_barrier`.

### Measured breakdown (post-Perf-1 op)
Cumulative ablation on the focus shape, bench `test_perf1_breakdown.py`.

| variant | ns |
|---|---|
| full | 157 057 |
| read + write (compute stubbed) | 159 214 |
| read + compute | 93 144 |
| write + compute | 112 533 |
| read only | 83 334 |
| write only | 110 139 |
| compute only | 40 416 |
| sync floor | 1 174 |

- **Ranked bottleneck**:
  1. DRAM/NoC data movement. Compute is fully hidden: full ≈ DM-only. The combined rate is ~427 GB/s against a
     ~512 GB/s peak, and the Perf 1 DM ceiling is 131–142 µs.
  2. Within DM, the writer is the weaker half: 305 GB/s alone, vs 404 GB/s for the reader.
  3. Compute is off the critical path. No compute ideas were floated.

### Portfolio and verdicts
Each experiment's directory is `perf_experiments/<slug>/` and its test is
`tests/.../test_perfexp_<slug>.py`.

| idea | verdict | focus before → after | domain / notes |
|---|---|---|---|
| `posted_stateful_dm`: posted writes with a fence tail; stateful one-packet write; reader twin: stateful one-packet read | **NULL** | 159.3 → 156.8–159.8 µs (±1%); writer-only 110.2 → 109.2–111.3 | Every combination is flat on all shapes and bit-exact. Per-transaction issue cost is not the limit: about 470 ns per tile per core, set by DRAM contention and skew. |
| `trid_pipelined_dm`: trid-scoped read barrier and write flush, T = 2/3/4 | **NULL** | 158.3 → 157.6–159.8 µs | Flat at full grid: 120 cores × 8 tiles in flight already saturate DRAM. Measured regression of +40–60 ns at 1 tile/core. Bit-exact on 9 shapes. |
| `bank_affine_split`: each core owns pages from the NoC-nearest DRAM bank group (banks grouped by anchor column), counts unchanged | **WIN** | 157.7 → 151.7 µs (subagent medians); DM-only 158.0 → 145.9 | Bank groups come from `get_optimal_dram_bank_to_logical_worker_assignment(NOC_0/1)`; nothing is fitted to this board. Controls: far column is +13%; 1 or 2 banks per core is +21%, because a core loses bank parallelism. Bit-exact on 10 shapes, including ragged ones. |
| `dynamic_claim_v2`: a dispatcher core hands out block ids via semaphore mailboxes, with no fetch-and-add | WIN, **not graduated** | 157.8 → 153.3 µs | Correct over about 14 400 launches. Root cause of Perf 1's duplicates: the INCR_GET return word is unreliable when it is polled. Measured regressions: (2,4,256,512) +3 to +10% and (1,1,32,32) +12%. It costs a core plus 4 kernels and 5 semaphores. It conflicts with `bank_affine_split`, since both decide page ownership, and loses to it on every shape. |

### Graduated
**`bank_affine_split`** is now the op's only work-split path. The contiguous `start_tile` page ranges were
deleted from the descriptor, reader and writer.
- The host keeps `split_work_to_cores`' per-core tile counts, so the CBs, the compute kernel and its runtime args
  are unchanged.
- The host assigns each core a short segment list over the bank groups with a greedy, regret-ordered
  transportation fill.
- The reader and writer walk that same list, with no device divide.
- The runtime-arg length is fixed at `3 + n_groups·(4 + group_max)`, so a program-cache hit cannot index past
  the args.
- Raw LLK: none new. The kernels still use `noc_async_{read,write}_page` through the `TensorAccessor`.
- **Carve-outs: none.** The ≤1 tile/core regime measures +50 to +85 ns. A same-session attribution run
  (`test_perf2_small_regime.py`) pushed the old contiguous assignment through the new kernels and was no faster:
  1.30 vs 1.31 µs, 1.38 vs 1.39 µs, 1.53 vs 1.48 µs. So the cost is fixed kernel overhead (the segment walker and
  19 runtime args instead of 3), not routing.
  - A host-side assignment switch would not recover it; only a second kernel path would.
  - I judged at most 85 ns on a ~1.3 µs launch not worth a dual path. The cost is recorded here as a measured
    small-shape regression rather than fenced off.

### Whole-op before → after, guard set
Same session, same bench. Before is the HEAD (Perf 1) kernels and descriptor; each value is the mean of 2 runs.

| shape | tiles | before ns | after ns | Δ |
|---|---|---|---|---|
| (1,1,4096,4096) **focus** | 16 384 | 158 878 | 147 233 | **−7.3%** |
| (8,1,1024,1024) | 8 192 | 80 716 | 74 272 | −8.0% |
| (2,4,256,512) | 1 024 | 11 886 | 10 329 | −13.1% |
| (1,1,32,8192) | 256 | 3 316 | 3 223 | −2.8% |
| (1,1,256,256) | 64 | 1 691 | 1 671 | flat |
| (2,64,256) | 16 | 1 428 | 1 475 | +47 ns (+3.3%) |
| (64,128) | 8 | 1 256 | 1 340 | +84 ns (+6.7%) |
| (1,1,32,32) | 1 | 1 261 | 1 327 | +66 ns (noisy, ±60 ns) |

- The data-movement-only path (compute stubbed) is faster on every shape: the focus shape goes from 157.4 to
  146.2 µs.
- Golden `eval/golden_tests/clamp_square_practice/`: 46/46. Op unit, extended and precision-baseline tests:
  68/68. Output is bit-exact against the reference.

### Helper bypasses — none
No raw-LLK path was graduated in this round. The Perf 1 `ClampSquareFused` bypass still stands, unchanged.

**Summary**: 4 ideas measured.
- 1 graduated: the bank-affine work split, as the op's one path, with no carve-out.
- 2 null: posted/stateful DM and trid-pipelined DM.
- 1 correct win not graduated: dynamic claim. It conflicts with the graduated split and is beaten by it.

The op is 7.3% faster on the focus shape and 3–13% faster on mid and large shapes. The ≤1 tile/core shapes are
50–85 ns slower; that cost is measured, attributed to kernel overhead and left uncarved.
