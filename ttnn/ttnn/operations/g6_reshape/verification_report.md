# Verification Report: g6_reshape

Board: Blackhole p150, core grid overridden to 11×10 = 110 cores (`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`),
DRAM alignment 64 B. Phase 0 scope: `dtype = bfloat16`, `layout = TILE_LAYOUT`, DRAM interleaved, every
rank 1..4 on both sides, both tile regimes (`tile_face_row`, `tile_element_run`).

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | **DRY, L1 sizing.** The BB-05 cap on `block_atoms` used a hand-written `bytes_per_trs` sum; the CB sizes used three separate per-CB formulas, and `l1_bytes_per_core` used a third expression. A change to one would quietly move the cap off the real allocation. | One helper `_slot_bytes(trs, S, F, A)` gives the three per-mover terms, `_core_bytes()` folds in movers and depths, and `_max_copies()` gives copy-entry capacity (also a CT arg). The cap, CB sizes and `describe_plan()["l1_bytes_per_core"]` all derive from these. |
| 2 | **DRY, mover placement.** Mover → RISC/NoC was stated twice: `MOVER_ROLES` strings in the plan report, and `Reader/WriterConfigDescriptor()` selected by `mover == 0` in the builder. | One `MOVERS` table `(cb ids, DataMovementProcessor, NOC)` feeds the builder, through an explicit `DataMovementConfigDescriptor`, and the plan report. Placement is unchanged: RISCV_1/NOC_0 and RISCV_0/NOC_1, default O2 build (`program.cpp:576-586`). |
| 3 | **Blocking-model fidelity: `block_atoms` had no knob.** The design names it a host knob, but it could only be reached indirectly through `L1_OP_BUDGET`. | Added `MAX_BLOCK_ATOMS` (default `None` = coarsest block that fits, so default behaviour is unchanged). It is clamped before the rounding fit loop. |
| 4 | **BB-07 cache key incomplete for knobs.** Module knobs (`MAX_CORES`, `MOVERS_PER_CORE`, `STAGE_DEPTH`, `IMAGE_DEPTH`, `L1_OP_BUDGET`) change the program but were not in `_cache_key`. A knob turned at runtime (perf tuning, tests) replayed the stale descriptor. | `_knobs()` is part of the key. |
| 5 | Dead constant `TILE_BYTES_PER_ELEM`. | Removed. |
| 6 | Kernel: staging and copy-entry capacity had `--dev` ASSERTs; the tile table did not. | `ASSERT(num_tiles < kBlockAtoms)` (an atom never straddles an output tile). |
| 7 | Plan report could not show whether a call runs more than one block per mover. | `describe_plan()["max_blocks_per_mover"]`. |
| 8 | **Untested code path.** Every golden shape is one block per mover on the full grid, so the block loop, the stage/image slot rotation (`STAGE_DEPTH`/`IMAGE_DEPTH = 2`), a single mover and a single core had never run. | `test_block_knobs`: 4 knob settings × 3 production shapes (both regimes), each forcing > 1 block per mover, with NaN-poisoned padding. 12/12 pass in plain and `--dev` (NoC sanitizer + kernel ASSERTs). |

Reviewed and correct as written: DRAM read congruence (each run's staging address ≡ source NoC address mod 64)
and write congruence (image regions at multiples of `F`); ownership atoms at every core/mover/block
boundary; the 64-bit stream divide once per (segment, half) with incremental steps per row (the carry
bound `in_h + step + 1 < 2·H_in` holds); the element-run piece walk stops at `W_in`, so padding columns
never reach a logical output; `noc_async_writes_flushed` before image reuse, `invalidate_l1_cache` after
the read barrier, a `load_blocking` store drain before the NoC reads the image; `kernel_main`,
`api/dataflow/dataflow_api.h`, `TensorAccessor` (no `InterleavedAddrGen`); no CB handshakes (private scratch, push = wait = 0).
Profiler zones cover every block operation (BB-21). The `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` build also passes the 16
production/element-run acceptance cases (BB-04).

Helper usage: no compute phase exists. The design's rejection table holds: `copy_via_memmove` pays an
invalidate and drain per call, `tt_memmove` needs 16-B congruence that element-run pieces violate, and
`copy_tile_column` has the wrong shape. No inter-core traffic exists, so the mcast helper does not apply.

Deferred: none. Perf observations are below, under Recommendations.

### Prompt rules (`eval/prompts/g6_reshape.txt` `## Rules`)

| Rule (applies now) | Status |
|--------------------|--------|
| Bit-exact, no conversion | ✓ values move by NoC and RISC-V word/halfword copies only; precision baseline 0 ULP |
| No input padding into logical outputs | ✓ golden NaN-poison loose cases, acceptance poison tests, `test_block_knobs` poison |
| No staging through DRAM or host | ✓ the only allocation is the output tensor; no host reads |
| Cores = blocking choice | ✓ `P = min(110, G)` atoms; `(16,1,64)→(1,1,16,64)` (2 output tiles) runs on 16 cores |
| No host-side transforms | ✓ |
| OOM never refused or excluded | ✓ `block_atoms` caps; multi-block path verified |
| dtype + production layout in one change | re-arms in Refinement 1 |
| BB-01 one dispatch | ✓ one `generic_op`; `device_num_programs = 1` per call |
| BB-03 NoC drained | ✓ read barrier per block, write barrier at the end |
| BB-05 L1 ≤ 1347776 | ✓ measured peak ≤ 12672 B per core |
| BB-06 repeat identical | ✓ `test_repeat_bit_identical`, `test_retarget_and_repeat` |
| BB-07 cache | ✓ (now including the knobs) |
| BB-08 plan report | ✓ `describe_plan` |
| BB-16 trace | ✓ with program cache on (see harness finding H1) |
| BB-17 few core ranges | ✓ ≤ 2 rectangles |
| BB-18 default opt level | ✓ O2 |
| BB-21 zones + roles | ✓ |
| BB-22 test scope | ✓ production shapes. The acceptance file's bf16 element-run cases pin the regime Phase 0 claims, and the knob test uses production shapes. |

## Registry Conformance

- `INPUT_TAGGERS = {"dtype": tag_dtype, "layout": tag_layout}`, both `(inputs, axes)`. Axis names and values match `feature_spec.TARGET`.
- `SUPPORTED = {"dtype": [bfloat16], "layout": [TILE_LAYOUT]}`. Every axis the kernel gates on is present. Placement is not a TARGET axis and is guarded separately in `validate()`.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the placement guard, then the shape checks (`ValueError`). `g6_reshape()` calls it on its first line.
- The op file declares no `INVALID`.
- No XPASS evidence, so no SUPPORTED auto-fix. Probe `probes/probe_007.py` (`validate()` bypassed) ran 10/10 bit-exact for fp32/int32/uint32 TILE, so the tile kernel already moves 4-byte elements. SUPPORTED is left at the Phase 0 contract, and the finding is folded into Refinement 1.
- **INVALID audit** (`feature_spec.py`): `INVALID = []` is correct. TARGET holds no block-float dtype, so the canonical bf8b+ROW_MAJOR entry does not apply. The op is single-tensor, so no cross-tensor coupling exists. There are no weight axes.

## Design Conformance

- **Algorithm**: output-stationary gather → await → permute → store, face-row granule, both regimes selected by the exact predicate. ✓
- **Topology**: one kernel source instantiated as two movers per core, each on its own NoC; no compute kernel. ✓
- **Distribution**: column-major atom fold, `P = min(grid, G)`, contiguous atom ranges, mover split ⌈n/2⌉/⌊n/2⌋. The large cells fill all 110 cores. Small cells are limited by atoms, as designed: `(8,1,64)` has 8 atoms on 8 cores; one mover per core idles when a core owns a single atom. ✓
- **Inter-core**: none. ✓
- **Blocking fidelity**: `MAX_CORES`, `MOVERS_PER_CORE`, `MAX_BLOCK_ATOMS` (added), `STAGE_DEPTH`, `IMAGE_DEPTH` are parameters, each with one source. Every CB size, trip count and RT range derives from them. No CB scales with a whole-op dimension.
- **Expression**: each block operation is block-scoped. Gather issues every read of the block (coalesced runs), then one barrier. Permute runs one flush/invalidate/drain per block. Store issues merged runs with no per-unit wait, and the write barrier runs once per kernel. No per-unit handshake in any role.

## L1 Ledger Audit

- **Currency**: 6 CBs declared, 6 ledger rows. Capacity expressions now carry the `align_up(·, 64)` the code applies (fixed). The symbol table names the knobs.
- **Capacity vs live set**: staging exceeds its live set only by the per-run phase pad (`< A`, DRAM read congruence) and by `S` in element-run. Both are stated mechanism reasons. Image and plan equal their live sets. No under-scaling: each buffer that spans `out_atom`/`half` scales with `T`.
- **Page format vs DEST**: not applicable. All payloads are dataflow-only, with no DEST and no `fp32_dest_acc_en`.
- **Disjoint lifetimes**: every pair has a stated concurrency reason (permute reads staging while writing the image; the tile table is read during store).
- **Bounds**: `S ≤ 16`, `atom_rows ∈ {1,2}`, and `block_atoms` is BB-05-capped. The total is closed-form.
- **Data-movement budget**: present and consistent with the code. Each input face-row crosses DRAM once in `tile_face_row` (≤ 2× on straddling face-rows in element-run), each output face-row once, with zero cross-core traffic. The cheapest-traffic split (A, column-major output-atom fold) is the one implemented.
- **Block-size defaults**: held. The work spreads over the full grid first, then the coarsest block that fits, which is one block per mover on every golden shape. No departure, so no measurement is owed.
- **Footprint**: `L1/core ≈ movers · T · (stage_depth·2S(F+A) + image_depth·2F + 16S + 16)`. Staging scales with `block_atoms`, `S`, `stage_depth`; image with `block_atoms`, dtype, `image_depth`; plan with `block_atoms`, `S`. Measured graph-capture peaks equal the formula: 1152–12672 B per core.

No ledger finding needs a refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_reshape/test_g6_reshape_precision_baseline.py` (randn bf16, seed 7):

| Shape | Regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true median (p5, p95) |
|-------|--------|-----|-------------|--------------|------------------|---------|---------------------------|
| (16,1,64)→(1,1,16,64) | tile_face_row | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| (1,16,6,256)→(1,1,16,1536) | tile_face_row | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| (16,1,5120)→(1,1,16,5120) | tile_face_row | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| (1,185)→(1,1,185,1) | tile_element_run | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |

**Assessment**: bit-exact. Every bit pattern is preserved, including NaN, ±inf, −0.0, denormals and odd last
mantissa bits (golden special-value and poison cases, acceptance `test_special_values`). No scale signature exists:
the ratio is exactly 1.

**Recommended tolerances**: exact for every dtype, i.e. `helpers.TOLERANCES = ("exact", None)`. No PCC/RMS limit applies.

## Verifier CLI Summary

Golden suite run twice on the final code. Reports: `verifier_report.json` (graded default,
`EVAL_CAPTURE_L1=1`) and `verifier_report_no_l1_capture.json` (`EVAL_CAPTURE_L1=0`), both in this directory.

| Category | Graded default (L1 capture on) | L1 capture off |
|----------|-------------------------------|----------------|
| supported_pass | 20 | 21 |
| xfail_expected | 32 | 32 |
| invalid_skipped | 0 | 0 |
| supported_fail | **1** (harness H1) | 0 |
| xpass_drift | 0 | 0 |
| xfail_wrong_mode | 0 | 0 |
| no_axes_found (test_regression.py) | 6 (2 pass, 4 harness H2) | 6 (6 pass) |

The 32 `xfail_expected` cells are exactly TARGET − SUPPORTED: 20 int32/uint32 ROW_MAJOR golden cells, 5 fp32 TILE
golden cells, and 7 loose cases (the flagged `[1,4096]→[4096,1]` int32 ×2, its `-1` variant, the int32 `[16]`
sequence, uint32 high bit, fp32 special values, fp32 NaN-poisoned padding). Refinement 1 moves them all.

### Harness findings (not op defects; nothing in the op can fix them)

- **H1. The trace loose case cannot pass under graded L1 capture.** `eval/l1_profiling.py:start_for_item` runs
  `device.disable_and_clear_program_cache()` for every test. The loose case's eager call therefore never caches
  its program, and `begin_trace_capture` → `generic_op` hits `TT_FATAL mesh_workload.cpp:617 "Cannot load new binaries
  during trace capture ... not yet in program cache"`. This holds for every op that dispatches a program. With
  `EVAL_CAPTURE_L1=0` the same cell passes, including replay bit-equal to eager. Fix belongs in the harness:
  skip L1 capture for `extras["trace"]` cells, or keep the cache enabled for them.
- **H2. `test_regression.py::test_validation_errors[*]` (4 tests) fail under L1 capture** with
  `L1ProfilingError: peak_total=0`. The op correctly raises `ValueError` in `validate()` before any device
  allocation, so a correct op has no L1 peak to report. The capture should skip tests that never dispatch. All 4 pass with
  `EVAL_CAPTURE_L1=0`.

## Recommendations

- **Refinement queue** (`op_requirements.md`): R1 is the only generality refinement. It covers int32 + uint32 + float32 +
  ROW_MAJOR, via the design's deferred `rm_row_gather` regime plus the already-working 4-byte tile path. R2 and R3 are perf
  refinements on the flagged int32 row-to-column case.
- **Perf observation, bf16 TILE path.** It is RISC-V-cycle-bound, not DRAM-bound. Zone profile of
  `(16,1,5120)→(1,1,16,5120)` (`probes/probe_008.py`, 3 runs, 110 cores, ~24 output face-rows per mover):
  gather 2.0–2.4 µs, await 1.8–2.5 µs, permute ≈ 1.5 µs, store ≈ 0.36 µs, kernel span ≈ 8.7 µs for 160 KB moved.
  `(1,1,16,4608)→(16,1,4608)`: gather ≈ 2.4–2.6 µs, permute ≈ 1.4 µs, store ≈ 1.7–1.8 µs, await ≈ 0.3 µs, span ≈ 7.1 µs.
  That is roughly 250 ns of RISC-V time per output face-row, spread over the walk, the plan written to L1 and read back, and
  the copy. The design's "store realization" lamp (write face-rows straight from staging in `tile_face_row`, no permute
  and no image) and "copy engine" lamp target exactly this. Per the queue rules, perf refinements target the flagged
  loose case. These cells are guard-set members, and the R3 levers act on the same mover skeleton.
- Graded golden device-ns (L1 capture off, profiler build), bf16 TILE:
  `(16,1,5120)` 8781, `(1,1,16,4608)` 7015, `(8,1,5120)` 5198, `(1,1,8,4608)` 4607,
  `(16,1,1536)` 3716, `(1,16,6,256)` 3629, `(2,1,5120)` 2801, `(8,1,1536)` 2504, `(1,8,6,256)` 2467,
  `(8,1,64)` 2227, `(16,1,64)` 2147, `(1,16,64)` 2046, `(1,8,64)` 1975, `(1,1,6,256)` 1951 ns. The ~2 µs floor on 8–48
  cores is launch plus one DRAM round trip.
- Knobs are module-wide, so a perf tune for the RM regime moves the tile regimes too. If the regimes want
  different `MAX_CORES`/`MOVERS_PER_CORE`, make the knobs per-regime. They are already in the cache key.
- Environment: the slot's `ttnn` import needs `LD_PRELOAD` of the MPIX stub from `.tt_chip_scope`. Plain
  `python3 -m eval.verify_supported` fails without `source .tt_chip_scope`.
