# Verification Report: multiply

Target: Quasar (`quasar`) on the ttsim simulator. Specification: `eval/specs/multiply/spec.py` (sha256
`c3df857d65b7…`). Golden results: `/tmp/multiply_golden_v2/` (`verifier_report.json` there).

## Code Review

Fixed in place:

1. **The compute config was not passed through whole.** The host built `ComputeHardwareConfig` from
   `math_fidelity` and `fp32_dest_acc_en` only, so it silently dropped `math_approx_mode` and
   `dst_full_sync_en`. GUIDANCE says to pass the caller's config as-is. A new `compute_hardware_config(cfg)` in
   `multiply_program_artifacts.py` now maps:
   - `math_approx_mode` → `sfpu_precision_mode` (Approximate / Precise);
   - `dst_full_sync_en` → `double_buffer_dest = not dst_full_sync_en`;
   - `fpu_math_fidelity` and `enable_32_bit_dest` as before.

   The defaults are unchanged: Precise, double-buffered DEST. New test `test_compute_config_passed_through`
   checks that HiFi4 + approx + full-sync is exact and that LoFi is accepted (rule R4), and reads the fields
   back off the program spec.
2. **`block_cols_cap` used one page size for every ring.** It sized the b-window ring and both stream rings
   with `out.buffer_page_size()`. That is correct only while a, b and out share a dtype. It becomes wrong as
   soon as Refinement 4 brings mixed dtypes (bf16 operands → float32 output would under-size the b ring's
   bound). It now takes `a`, `b` and `out` page sizes separately. `l1_ledger.md`'s symbol table was updated
   to match.

Checked, no change needed:

- One `ttnn.generic_op` dispatch per call. The regime is chosen on the host and passed as a CT arg.
- Kernels use `void kernel_main()`, `api/...` includes, `TensorAccessor` and the Metal 2.0 NoC/DFB API.
- Every ring is implicit sync, except the `row_bcast_strips` b window. That ring is explicit sync, listed in
  `disable_dfb_implicit_sync_for`, and uses user trid 0 with a barrier on the same id.
- Both data-movement kernels hand the tag back as their last statement.
- Push/wait counts agree: the b window pushes, waits and pops a nominal `block_cols` per block on both sides.
  a and out move one entry per tile in the walker's shared order.
- Compute uses the library `mul` helper in both regimes, with no raw LLK:
  - `flat_stream`: Scalar/PerTile;
  - `row_bcast_strips`: `BroadcastDim::Row` + `InputTileMapping::Row` on the held window.

  Broadcast dimension: b's valid region is row 0 and the helper uses Row (srcB row 0 → 32 rows). The reader
  reads b once per block, with no redundant fill.
- The pack-target gap does not bite: every helper call packs into the same `out` ring.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification. The op file defines no axis function.
- `SUPPORTED` equals `PHASE0` on all nine axes.
- `EXCLUSIONS = []`.
- `validate()` derives `memory_config` / `sharding` with `eval.memory.memory_axes` for each operand (R10).
  It then runs these checks, in this order:
  1. R1 / R2 refusals;
  2. SUPPORTED, per axis;
  3. EXCLUSIONS.

  Every refusal raises `UnsupportedAxisValue` / `ExcludedCell`.
- X1 (shapes that do not broadcast) raises `ValueError` before the axes are tagged.
- `multiply()` calls `validate()` as its first line.
- `default_compute_kernel_config()` is exported, and it is the only place the default is written (R3).
- `validate()` reads neither `math_fidelity` nor `math_approx_mode` (R4).
- The op file declares no `INVALID`.
- `python3 -m eval.validate_contract multiply --phase0 --text`: PASSED, 0 errors. Its 5 "unresolved" items are
  the spec's own UNRESOLVED markers, not defects.
- No XPASS drift, so nothing was auto-added to SUPPORTED.

**INVALID audit (specification)**: `INVALID = []`, which is correct.
- TARGET has no block-float dtype, so the canonical bf8b + ROW_MAJOR entry does not apply. Quasar has no
  bfloat8_b path, and the target page says not to list one.
- No cross-tensor coupling.
- multiply is not norm-like, so no weight-canonicalization cells are needed.
- R1 / R2 (float32 with `fp32_dest_acc_en=False`) are correctly expressed as refusals, not INVALID: the cells
  are legal but lossy.

**Rules R1–R10**:
- R1, R2, R3, R4, R6, R7, R8, R10: met.
- R5 (either operand order) re-arms when `first_operand` enters SUPPORTED (Refinement 2).
- R9 (output in a's buffer, interleaved) is tested by `test_output_placement_follows_first_operand`.

**Specification defects**: none found.

## Design Conformance

- **Algorithm / pipeline / RISC ownership**: as designed. Two regimes run in one ProgramSpec, and the shared
  block walker header (`multiply_block_walker.hpp`) is the single source of the decomposition for all three
  kernels.
- **Work distribution**: `split_work_to_cores(..., row_wise=True)` over every cluster.
  - `flat_stream` splits tiles.
  - `row_bcast_strips` splits (n, g, c) strips, with `G = clamp(ceil(32 / (N_out·Wt)), 1, Ht)` row groups only
    when the strips under-fill the grid.
  - Plan values match the design's worked figures: `[1,4,8400]` → 7×9 + 25×8 units over 32 clusters;
    `[128,512]×[1,512]` → G = 2. Both are pinned by `test_regime_pinned`.
  - Small calls (e.g. `[1,4,400]`, 13 tiles) engage only the clusters that have work, which is correct.
- **Thread plan**: `THREAD_PLAN = (1, 1, 1)`, one named constant passed as CT args. The host program carries it
  as `num_threads` on each kernel. The design states the op as bound by data movement, with its reason: one
  FPU op against three tile transfers per tile, and toy_thread_plan's figures for this exact multiply.
  - The required tests exist and pass:
    - the plan is carried and the work covers the output;
    - `flat_stream` runs at (2,2,2), (4,4,2) and (1,4,1);
    - one device session with a device tilize before and `ttnn.to_layout(…, ROW_MAJOR)` after, in both
      regimes.
  - This backend has no timing model, so the plan alternatives cannot be measured here (backend page). The
    design's own lamp (measure (2,2,2) and (4,4,2) on ttsim-perf) stands as a recommendation.
- **Blocking-model fidelity**: knobs are defined once on the host and every ring capacity and CT/RT arg
  derives from them: `THREAD_PLAN`, `STREAM_DEPTH`, `B_WINDOW_DEPTH`, `L1_RESERVE_BYTES`, and the per-call
  `G` and `block_cols`.
  - No CB scales with a whole-op dimension. The b window scales with `block_cols`, which is clamped by the ring
    cap and the budget.
  - Per-core compute is coarse: one helper call per core in `flat_stream`, one per block in
    `row_bcast_strips`. No half-turned split.
  - Expression: reader and writer issue one tagged transfer per tile under implicit sync. That is the target's
    protocol (DM0 posts credits), not a per-unit handshake. The b window is one barrier per block.
- **Inter-core communication**: none, as designed.

## L1 Ledger Audit

- **Currency**: three rings (`in_a`, `in_b` [stream or window], `out`). Each has a row, and each size
  expression matches the code. The only edit was the `block_cols_cap` expression (fix 2 above).
- **Capacity vs live set**:
  - `in_a`, `out` and the stream `in_b`: capacity 4 against a live set of 1. Justified by implicit sync's two
    txn-id halves (mechanism reason).
  - b window: `2·block_cols` against a live set of `block_cols`. Justified by prefetch plus the liveness
    requirement.
  - No live set spans an axis its capacity does not scale with. The window spans w through `block_cols`; h
    and n are streamed.
  - Note for Refinement 3: on `Ht = 1` shapes the window is held but never reused (block_rows = 1). That is
    a schedule cost rather than an L1 one, and it is folded into Refinement 3.
- **Page format vs DEST**:
  - Inputs are bf16 and unpack to SrcA/SrcB (they never pass through DEST).
  - The output page is bf16 under 32-bit DEST: the *under* case, answered by the output ABI. The output is
    bf16 and R6 asks for one rounding.
  - No Float32 page under 16-bit DEST.
- **Disjoint lifetimes**: none. All rings are live concurrently.
- **Bounds**: every capacity symbol has a bound and the predicate that sets it, and the total is closed-form.
- **Footprint**:
  - `flat_stream`: `3·STREAM_DEPTH·C·2048` = 24,576 B.
  - `row_bcast_strips`: `2·STREAM_DEPTH·C·2048 + B_WINDOW_DEPTH·block_cols·C·2048` = 16,384 + 4,096·block_cols B.
    That is 151,552 B on `[1,4,33600]`, and at most 1,060,864 B at the 255-tile cap, out of 3,963,328 B.
  - Only `block_cols` (w) and the depth knobs scale the footprint.
- **Data-movement budget**: present, and consistent with the code. a and out cross once; b crosses G times in
  `row_bcast_strips` and once in `flat_stream`.
  - The cheapest split, a multicast of b to the G row groups, is a `deferred` row with a positive reason: G > 1
    only when `N_out·Wt < num_cores`, so it saves fewer than `num_cores` tile reads per call.
  - Once Refinement 2 lands `over_leading` through strides, b will cross `N_out/N_b` times. The design's
    `leading_reuse` row covers that, and Refinement 5 is the perf entry that removes it.
- **Block-size defaults**: interleaved work spreads across the full grid, then takes the coarsest block. In
  `row_bcast_strips`, `block_cols = min(Wt, max_units_per_core, cap)`, so one block per core on every YOLO
  shape. Held; no departure.

## Precision Baseline

`tests/ttnn/unit_tests/operations/multiply/test_multiply_precision_baseline.py`: bf16 operands drawn from a
standard normal, default config (HiFi4, fp32 DEST). Errors are measured against the float64 product of the
stored inputs. ULP is measured against that product rounded once into bf16.

| Shape (a × b) | Mem | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP max / p99 | Bit-exact | got/true median [p5, p95] |
|---|---|---|---|---|---|---|---|---|
| 32×17 × 1×17 | DRAM | 0.9999988 | 0.0127 | 7.5e-4 | 1.53e-3 | 0 / 0 | 100% | 1.0000 [0.9976, 1.0025] |
| 1×1×32×64 (equal) | DRAM | 0.9999987 | 0.0151 | 9.0e-4 | 1.61e-3 | 0 / 0 | 100% | 1.0000 [0.9974, 1.0026] |
| 1×4×400 × 1×1×400 | L1 | 0.9999986 | 0.0139 | 9.3e-4 | 1.65e-3 | 0 / 0 | 100% | 1.0000 [0.9973, 1.0025] |
| 1×4×8400 × 1×1×8400 | DRAM | 0.9999986 | 0.0303 | 9.0e-4 | 1.66e-3 | 0 / 0 | 100% | 1.0000 [0.9973, 1.0026] |
| 2×4×128×512 × 2×4×1×512 | DRAM | 0.9999986 | 0.0312 | 8.8e-4 | 1.65e-3 | 0 / 0 | 100% | 1.0000 [0.9974, 1.0026] |
| 2×512×1024 (equal) | DRAM | 0.9999986 | 0.0312 | 8.9e-4 | 1.65e-3 | 0 / 0 | 100% | 1.0000 [0.9974, 1.0026] |

**Assessment**:
- Every element equals the exact product rounded once to bf16. The FPU product of two bf16 values is exact
  in fp32 DEST at HiFi4, and the packer rounds to nearest even.
- The residual error against float64 is that single bf16 rounding: relative RMS ≈ 1.6e-3, and the ratio
  spread is ±0.26%, half a bf16 ulp, centred on 1.0. No scale signature.

**Recommended tolerances**: exact. Every cell resolves in `NUMERIC_RESOLVED` to 0 ulp, and Phase 0 meets it.

## Verifier CLI Summary

Run after the fixes (`/tmp/multiply_golden_v2`): 1640 tests, 99 passed, 0 failed, 0 hangs.
- supported_pass: 98 (92 PHASE0 cells + 6 numeric-regression cases)
- xfail_expected: 1540
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 2. `test_rejection[X1]` passed. `test_op_loose[NOTSET]` was skipped because LOOSE_CASES is
  empty.

The 1540 `xfail_expected` cells, by value that SUPPORTED lacks (a cell can lack several):

| Value | Cells (any) | Cells (blocked by it alone) | Refinement |
|---|---|---|---|
| `layout=ROW_MAJOR` | 816 | 92 | 1 |
| `broadcast` over_w / over_hw / over_leading / first_operand | 192 / 96 / 144 / 96 | 16 / 8 / 12 / 8 | 2 |
| `dtype=float32` | 816 | 92 | 4 |
| `fp32_dest_acc_en=False` | 816 | 92 | 4 |
| `output_dtype=float32` | 544 | 46 | 4 |

`verify_supported --queue op_requirements.md`: 8 values in TARGET − SUPPORTED, 0 queue gaps.

Tests in the op's directory: 38/38 pass.
- acceptance: 13
- `test_multiply_regimes.py`: 17 + 2 new (compute-config pass-through)
- precision baseline: 6

## Recommendations

- **Queue order and why**:
  1. RM layout (tier 3), first.
  2. Broadcast regimes (tier 4: regime extensions with no cross-core traffic). They build once on both layouts.
  3. Perf: the YOLO shapes.
  4. Numeric (tier 5). It extends every regime and layout with the SFPU float32 path once.
  5. Perf: leading-dim b reuse.
  6. Perf: per-block fixed costs.

  No loose case is perf-flagged. The first perf target is the YOLO detect-head calls the first phase exists
  for, which are already fully in SUPPORTED, so the anchor rule puts no constraint on Refinements 1 and 2.
- **Risk for Refinement 4**: the 0-ulp grade at `fp32_dest_acc_en=False` (bf16 product written to 16-bit
  DEST) depends on that write rounding to nearest even. If it does not, those cells stay failing as
  `numerical-precision` and get reported, not excluded.
- **Thread plan on a timed backend**: on ttsim-perf, measure (2,2,2) and (4,4,2) on `flat_stream` and on the
  YOLO `row_bcast_strips` shapes; it is a one-constant change. ttsim gives no time, so it is not queued here.
- **Not queued, with reasons**:
  - Multicasting b across G > 1 row groups saves fewer than 32 tile reads per call (design, `deferred`).
  - `dual_in_tile_bcast` (both operands expanded inside a tile) has no INPUTS entry. Refinement 2 refuses it by
    name.
- **L1 / memory pressure**: none near the budget. The largest per-core footprint on any INPUTS entry is
  151,552 B. With L1-placed operands, the rings and the tensors share each cluster's L1. The spec's callers'
  L1 tensors are a few hundred tiles, and `L1_RESERVE_BYTES` is the knob if a refinement needs headroom.
