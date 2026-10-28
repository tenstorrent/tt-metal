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
