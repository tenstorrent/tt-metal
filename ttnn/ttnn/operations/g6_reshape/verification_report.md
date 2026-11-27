# Verification Report: g6_reshape

Board: Blackhole p150, core grid overridden to 11×10 = 110 cores (`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`),
DRAM alignment 64 B. Scope after this pass: `dtype ∈ {bfloat16, float32}`, `layout = TILE_LAYOUT`, DRAM
interleaved, every rank 1..4 on both sides, both tile regimes (`tile_face_row`, `tile_element_run`).

This is the second verifier pass on the Phase 0 build. The first pass is commit `d1d7e125cad`. Every claim
below was re-measured in this pass unless it is marked as carried over.

## Code Review

### Fixed in this pass

| # | Finding | Fix |
|---|---------|-----|
| 1 | **SUPPORTED understated the kernel (float32).** `validate()` refused `float32` TILE, yet the tile kernel moves 4-byte elements bit-exactly. Probe `probes/probe_007.py` (validate bypassed) was re-run this pass: 10/10 bit-exact. That covers every FLUX fp32 cell, fp32 special values, fp32 NaN-poisoned padding, int32 extremes and uint32 high bit. float32's production layout is TILE, which is already supported, so claiming it satisfies the spec rule "a dtype lands with its production layout" and runs ≥ 1 golden cell. | `SUPPORTED["dtype"] = [bfloat16, float32]`. Golden: 5 fp32 cells + 2 fp32 loose cases (special values, NaN poison) moved from xfail to pass. int32/uint32 stay refused: their production layout is ROW_MAJOR, which must land in the same change (Refinement 1). |
| 2 | **L1 ledger, capacity > live set with a wrong stated bound (plan CB).** The tile table was sized `T = block_atoms·atom_rows` entries, and the ledger claimed "≤ T tile entries". The real bound is `block_atoms`: an atom never straddles an output tile, so each tile entry holds ≥ 1 whole atom. The kernel's own `ASSERT(num_tiles < kBlockAtoms)` already encodes this. bf16 over-allocated the table 2×. | `_slot_bytes(atoms, atom_rows, …)` sizes the table by atoms. The BB-05 cap is now per atom (`bytes_per_atom`). Ledger rows, footprint formula and worked values are corrected, as is the design's CB table. Graph-capture L1 peaks now equal the host plan: `(1,1,16,4608)→(16,1,4608)` 12672 → 12416 B, `(16,1,5120)→(1,1,16,5120)` 6912 → 6784 B. |
| 3 | **Coverage gaps for the new claim.** The acceptance test, precision baseline and multi-block knob test were bf16-only. fp32 has one-row atoms (`atom_rows = 1`), so its block and mover boundaries fall on every row, a path bf16 never runs. | The acceptance helpers are now dtype-generic. New tests: `test_float32_cases` (5 FLUX production shapes, NaN-poisoned padding, regime pinned to `tile_element_run`, `num_cores > 1`) and `test_float32_special_values`. `test_block_knobs` gains the `(1,185)` fp32 case under all 4 knob settings. The precision baseline adds `(1,512)→(1,1,512,1)` fp32, with an fp32-aware ULP map and a raw-bit compare (the ordered map folds −0.0 onto +0.0). 57/57 pass in plain and `--dev` (NoC sanitizer + kernel ASSERTs). |

### Fixed in the first verifier pass (re-checked, still holds)

DRY L1 sizing: one `_slot_bytes` feeds the BB-05 cap, the CB sizes and `describe_plan`. DRY mover placement: one
`MOVERS` table. `MAX_BLOCK_ATOMS` knob. Module knobs are in the descriptor-cache key (BB-07). Tile-table `--dev`
ASSERT. `describe_plan()["max_blocks_per_mover"]`. `test_block_knobs`, which forces multi-block, depth-2,
one-mover and one-core paths.

### Reviewed and correct as written

- DRAM read congruence: each run's staging address ≡ its source NoC address (mod 64).
- DRAM write congruence: image regions sit at multiples of `F`, at DRAM offsets that are multiples of `F`.
- Ownership atoms align on every core, mover and block boundary. Atoms restart per output image, and a ragged last atom never splits a 64-B DRAM word between movers.
- The 64-bit stream divide runs once per (segment, half), with 32-bit incremental steps per row. The carry bound `in_h + step + 1 < 2·H_in` holds.
- The element-run piece walk stops at `W_in`, so padding columns never reach a logical output. Padding rows are never addressed.
- `S` bound: 1 when aligned, 3 when `W_in ≥ 16`, `ceil(15/W_in)+1` otherwise, re-derived. The staging bound `≤ copies·(F + A − 1)` holds because each copy entry opens at most one run.
- The plan word packs the image offset `<< 8` (fits in 24 bits, with L1 ≪ 16 MB) and the length in 8 bits (`F ≤ 64`, static_assert).
- Each block runs `noc_async_writes_flushed` before reusing the image, `invalidate_l1_cache` after the read barrier, and a `load_blocking` store drain before the NoC reads the image.
- No CB handshakes: private scratch, push = wait = 0. Kernel conventions are correct: `kernel_main`, `api/dataflow/dataflow_api.h`, `TensorAccessor` with no `InterleavedAddrGen`.
- On a descriptor-cache hit only the common RT addresses are patched (`test_memo_hit_uses_live_addresses`). Addresses are never CT.

Helper usage: there is no compute phase. The design's rejection table still holds. `copy_via_memmove` pays an
invalidate and drain per call, `tt_memmove` needs 16-B congruence that element-run pieces violate, and
`copy_tile_column` has the wrong shape. There is no inter-core traffic, so the mcast helper does not apply.

### Deferred (documented, not a defect on any golden shape)

- **`STAGE_DEPTH` / `IMAGE_DEPTH` are plumbed but inert.** They size the CBs and rotate slots, but the block loop is
  strictly serial: gather(k+1) issues after store(k), and `noc_async_writes_flushed` drains every write before
  each permute. So depth 2 costs L1 and buys no overlap. Making it pay needs software pipelining: gather(k+1) between
  await(k) and permute(k), and a depth-2 plan CB, because the plan of k is read by permute(k) and store(k). That only
  matters when a mover owns > 1 block, and no golden shape does (the coarsest block is all of a mover's atoms, at
  ≤ 12416 B per core). It is listed as an overlap lever in Refinement 3, not as a separate refinement.
- BB-21: per-mover setup (arg reads, accessor construction) sits outside a zone. It is a few dozen cycles before the
  first `g6.gather`, and every block operation and wait is zoned.

### Prompt rules (`eval/prompts/g6_reshape.txt` `## Rules`)

| Rule (applies now) | Status |
|--------------------|--------|
| Bit-exact, no conversion | ✓ values move only by NoC and RISC-V word/halfword copies. Precision baseline: 0 ULP on bf16 and fp32. fp32 special values (NaN, ±inf, −0.0, denormals, odd mantissa) are exact |
| No input padding into logical outputs | ✓ golden NaN-poison loose cases (bf16 ×3, fp32 ×1), acceptance poison tests (bf16 ×4, fp32 ×5), `test_block_knobs` poison |
| No staging through DRAM or host | ✓ the only allocation is the output tensor, and the host reads no tensor values |
| Cores = blocking choice | ✓ `P = min(110, G atoms)`. `(16,1,64)→(1,1,16,64)` (2 output tiles) runs on 16 cores, and fp32 `(1,W)→(1,1,W,1)` (1 output tile column) on 110 |
| No host-side transforms | ✓ |
| OOM never refused or excluded | ✓ `block_atoms` caps L1, and the multi-block path is verified for bf16 and fp32 |
| dtype + production layout in one change | ✓ float32 lands with TILE (already supported). Re-arms for int32/uint32 + ROW_MAJOR in Refinement 1 |
| BB-01 one dispatch | ✓ one `generic_op`, `device_num_programs = 1` per call |
| BB-03 NoC drained | ✓ read barrier per block, write barrier at the end |
| BB-05 L1 ≤ 1347776 | ✓ measured peak ≤ 12416 B per core |
| BB-06 repeat identical | ✓ `test_repeat_bit_identical`, `test_retarget_and_repeat` |
| BB-07 cache | ✓ keyed on shapes, resolved target, dtype, layout, memory config, device, grid and knobs |
| BB-08 plan report | ✓ `describe_plan` |
| BB-16 trace | ✓ with the program cache on (see H1) |
| BB-17 few core ranges | ✓ ≤ 2 rectangles |
| BB-18 default opt level | ✓ O2 |
| BB-21 zones + roles | ✓ (setup note above) |
| BB-22 test scope | ✓ production shapes and rule-required cases only |

## Registry Conformance

- `INPUT_TAGGERS = {"dtype": tag_dtype, "layout": tag_layout}`, both with the `(inputs, axes)` signature. Axis names and values match `feature_spec.TARGET`.
- `SUPPORTED = {"dtype": [bfloat16, float32], "layout": [TILE_LAYOUT]}`. Every axis the kernel gates on is present. Placement is not a TARGET axis and has its own guard in `validate()`.
- `EXCLUSIONS = []`.
- `validate()` checks, in order: SUPPORTED per axis (`UnsupportedAxisValue`), EXCLUSIONS (`ExcludedCell`), the placement guard, then the shape checks (`ValueError`). `g6_reshape()` calls it first.
- The op file declares no `INVALID`.
- SUPPORTED auto-fix: `float32` added, on probe + golden evidence (finding 1). This is not an XPASS case: `validate()` refused fp32, so the drift showed up as `xfail_expected`, not as `xpass_drift`.
- **INVALID audit** (`feature_spec.py`): `INVALID = []` is correct. TARGET holds no block-float dtype, so the canonical bf8b+ROW_MAJOR entry does not apply. The op is single-tensor, so no cross-tensor coupling exists, and it has no weight axes.

## Design Conformance

- **Algorithm**: output-stationary gather → await → permute → store, with a face-row granule. Both regimes are selected by the exact predicate. ✓
- **Topology**: one kernel source runs as two movers per core, each on its own NoC. There is no compute kernel. ✓
- **Distribution**: column-major atom fold, `P = min(grid, G)`, contiguous atom ranges per core, and a ⌈n/2⌉ / ⌊n/2⌋ mover split. Every cell with ≥ 110 atoms fills all 110 cores, all fp32 FLUX cells included. Smaller cells are limited by atoms, the design's indivisible ownership unit, not by tiles. For example, `(8,1,64)` has 8 atoms on 8 cores, and one mover per core idles when a core owns one atom. ✓
- **Inter-core**: none. ✓
- **Blocking fidelity**: `MAX_CORES`, `MOVERS_PER_CORE`, `MAX_BLOCK_ATOMS`, `STAGE_DEPTH` and `IMAGE_DEPTH` are parameters, each with one source. Every CB size, trip count and RT range derives from them. No CB scales with a whole-op dimension. The depth knobs are inert in the current schedule (see Deferred).
- **Expression**: each block operation is block-scoped. Gather issues every read of the block as coalesced runs, then one barrier. Permute runs one flush/invalidate/drain per block. Store issues merged runs with no per-unit wait. No role does a per-unit handshake.

## L1 Ledger Audit

- **Currency**: 6 CBs declared, 6 ledger rows. The plan-CB expression was wrong and is fixed (finding 2). Size expressions now match the allocation, and graph-capture peaks equal the host plan on every golden cell.
- **Capacity vs live set**: staging exceeds its live set only by the per-run phase pad (`< A`, DRAM read congruence) and by `S` in element-run. Both are mechanism reasons. The image equals its live set. The plan exceeds its live set only by `S` (element-run) and ragged atoms. No under-scaling: every buffer that spans `out_atom`/`half` scales with `block_atoms`.
- **Page format vs DEST**: n/a. Every payload is dataflow-only, and there is no DEST or `fp32_dest_acc_en`.
- **Disjoint lifetimes**: every pair has a stated concurrency reason. Permute reads staging while writing the image, and the tile table is read during store.
- **Bounds**: `S ≤ 16`, `atom_rows ∈ {1,2}`, and `block_atoms` is BB-05-capped and refused on the host if one atom does not fit. The total is closed-form.
- **Data-movement budget**: present and consistent with the code. In `tile_face_row`, each input face-row crosses DRAM once; in element-run, straddling face-rows cross ≤ 2×. Each output face-row crosses once, with zero cross-core traffic. The cheapest-traffic split (column-major output-atom fold) is the one implemented.
- **Block-size defaults**: held. The work spreads over the full grid, then takes the coarsest block that fits, which is one block per mover on every golden shape. There is no departure, so no measurement is owed.
- **Footprint**: `L1/core ≈ movers · block_atoms · (atom_rows·(stage_depth·2S(F+A) + image_depth·2F + 16S) + 16)`. Staging scales with `block_atoms`, `S` and `stage_depth`; the image with `block_atoms`, dtype and `image_depth`; the plan with `block_atoms` and `S`. Measured peaks: 1152–12416 B per core.

The ledger finding was fixed in place, so it needs no refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_reshape/test_g6_reshape_precision_baseline.py` (randn, seed 7):

| Shape | dtype | Regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true median (p5, p95) |
|-------|-------|--------|-----|-------------|--------------|------------------|---------|---------------------------|
| (16,1,64)→(1,1,16,64) | bf16 | tile_face_row | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| (1,16,6,256)→(1,1,16,1536) | bf16 | tile_face_row | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| (16,1,5120)→(1,1,16,5120) | bf16 | tile_face_row | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| (1,185)→(1,1,185,1) | bf16 | tile_element_run | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |
| (1,512)→(1,1,512,1) | fp32 | tile_element_run | 1.000000 | 0 | 0 | 0 | 0 | 1.0 (1.0, 1.0) |

**Assessment**: bit-exact. Every bit pattern is preserved, NaN payloads, ±inf, −0.0, denormals and odd last
mantissa bits included, for bf16 and fp32. The evidence: golden special-value and poison loose cases, acceptance
`test_special_values` and `test_float32_special_values`, and raw-bit compares everywhere. There is no scale
signature: the ratio is exactly 1.

**Recommended tolerances**: exact for every dtype. `helpers.TOLERANCES = ("exact", None)` already says so. No
PCC/RMS limit applies.

## Verifier CLI Summary

The golden suite was run on the final code in both L1-capture modes. Reports:
`verifier_report.json` (graded default, `EVAL_CAPTURE_L1=1`) and `verifier_report_no_l1_capture.json`
(`EVAL_CAPTURE_L1=0`), both in this directory.

| Category | Graded default (L1 capture on) | L1 capture off |
|----------|-------------------------------|----------------|
| supported_pass | 27 | 28 |
| xfail_expected | 25 | 25 |
| invalid_skipped | 0 | 0 |
| supported_fail | **1** (harness H1) | 0 |
| xpass_drift | 0 | 0 |
| xfail_wrong_mode | 0 | 0 |
| no_axes_found (test_regression.py) | 6 (2 pass, 4 harness H2) | 6 (6 pass) |

Before this pass (same code, fp32 still refused) the counts were 20/21 supported_pass and 32 xfail_expected. The 7
fp32 cells moved.

The 25 `xfail_expected` cells are exactly TARGET − SUPPORTED, and Refinement 1 moves all of them:
- 15 int32 ROW_MAJOR golden cells;
- 5 uint32 ROW_MAJOR golden cells;
- 5 loose cases: the flagged `[1,4096]→[4096,1]` int32 (plain and `int32_extremes`), its `[-1,1]` variant, the int32 `[16]` sequence, and uint32 high bit.

Golden device-ns (profiler build, graded / L1 off), cores, programs per call, graph-capture L1 peak:

| Cell | ns | cores | progs | L1 B |
|------|----|-------|-------|------|
| bf16 (16,1,5120)→(1,1,16,5120) | 8783 / 8964 | 110 | 1 | 6784 |
| bf16 (1,1,16,4608)→(16,1,4608) | 6974 / 6978 | 110 | 1 | 12416 |
| bf16 (8,1,5120)→(1,1,8,5120) | 5217 / 5309 | 110 | 1 | 3456 |
| bf16 (1,1,8,4608)→(8,1,4608) | 4604 / 4625 | 110 | 1 | 6784 |
| bf16 (16,1,1536)→(1,1,16,1536) | 3703 / 3767 | 110 | 1 | 2304 |
| bf16 (1,16,6,256)→(1,1,16,1536) | 3535 / 3562 | 110 | 1 | 2304 |
| bf16 (2,1,5120)→(1,1,2,5120) | 2686 / 2767 | 110 | 1 | 1152 |
| bf16 (8,1,1536)→(1,1,8,1536) | 2514 / 2510 | 110 | 1 | 1152 |
| bf16 (1,8,6,256)→(1,1,8,1536) | 2452 / 2475 | 110 | 1 | 1152 |
| bf16 (16,1,64)→(1,1,16,64) | 2130 / 2155 | 16 | 1 | 1152 |
| bf16 (1,16,64)→(16,1,64) | 2096 / 2079 | 32 | 1 | 1152 |
| bf16 (8,1,64)→(1,1,8,64) | 2070 / 2140 | 8 | 1 | 1152 |
| bf16 (1,8,64)→(8,1,64) | 2033 / 1973 | 16 | 1 | 1152 |
| bf16 (1,1,6,256)→(1,1,1536) | 1953 / 1959 | 48 | 1 | 1152 |
| fp32 (1,512)→(1,1,512,1) | 2175 / 2161 | 110 | 1 | 5760 |
| fp32 (1,256)→(1,1,256,1) | 1945 / 1959 | 110 | 1 | 3840 |
| fp32 (1,196)→(1,1,196,1) | 1804 / 1882 | 110 | 1 | 1920 |
| fp32 (1,185)→(1,1,185,1) | 2215 / 2156 | 110 | 1 | 1920 |
| fp32 (1,207)→(1,1,207,1) | 2147 / 2152 | 110 | 1 | 1920 |

### Harness findings (not op defects; nothing in the op can fix them)

Re-confirmed this pass, on the same nodes and with the same messages.

- **H1. The trace loose case cannot pass under graded L1 capture.** `eval/l1_profiling.py:start_for_item` runs
  `device.disable_and_clear_program_cache()` for every test. The loose case's eager call therefore never caches
  its program, and `begin_trace_capture` → `generic_op` hits `TT_FATAL mesh_workload.cpp:617 "Cannot load new binaries
  during trace capture ... not yet in program cache"`. Any op that dispatches a program hits the same failure. With
  `EVAL_CAPTURE_L1=0` the same cell passes, and its replay is bit-equal to the eager call. The fix belongs in the harness: skip L1
  capture for `extras["trace"]` cells, or keep the cache enabled for them. Not excluded, not gated: escalated.
- **H2. Under L1 capture, the 4 `test_regression.py::test_validation_errors[*]` tests fail** with
  `L1ProfilingError: peak_total=0`. The op correctly raises `ValueError` in `validate()` before any device
  allocation, so a correct op has no L1 peak to report. The capture should skip tests that never dispatch. All 4 pass with
  `EVAL_CAPTURE_L1=0`.

## Recommendations

- **Refinement queue** (`op_requirements.md`):
  - R1 is the only generality refinement left. It covers int32 + uint32 + ROW_MAJOR via the design's deferred `rm_row_gather` regime, and lands the whole perf-1 contract.
  - R2 and R3 are perf refinements on the flagged int32 row-to-column case. Once R1 lands, generality is exhausted, so every later slot is perf.
- **Perf observation, tile path (carried over from the first pass, `probes/probe_008.py` zone profile).** The mover is
  RISC-V-cycle-bound, not DRAM-bound: roughly 250 ns of RISC-V time per output face-row, split across the walk, the plan
  written to L1 and read back, and the copy.
  - `(16,1,5120)→(1,1,16,5120)`: gather 2.0–2.4 µs, await 1.8–2.5 µs, permute ≈ 1.5 µs, store ≈ 0.36 µs.
  - `(1,1,16,4608)→(16,1,4608)`: gather ≈ 2.5 µs, permute ≈ 1.4 µs, store ≈ 1.75 µs.

  The design's "store realization" and "copy engine" lamps target exactly this. Per the queue rules, perf refinements
  target the flagged loose case. These cells are guard-set members, and the R3 levers act on the same mover skeleton.
- The ~2 µs floor on small cells (8–48 cores, and the fp32 FLUX cells on 110 cores with 1–3 atoms per mover) is
  launch plus one DRAM round trip. The design's grid-synchronization lamp (fewer cores, more atoms per mover) is the
  measurable alternative. The fp32 FLUX cells (16 tiles max) and the four `*,64` bf16 cells (≤ 32 tiles) are outside
  what the perf rules let a perf pass measure, so they are guarded only by the golden suite.
- Knobs are module-wide, so a perf tune for the RM regime moves the tile regimes too. If the regimes want
  different `MAX_CORES`/`MOVERS_PER_CORE`, make the knobs per-regime. They are already in the cache key.
- Environment: the slot's `ttnn` import needs `LD_PRELOAD` of the MPIX stub from `.tt_chip_scope`. Plain
  `python3 -m eval.verify_supported` fails without `source .tt_chip_scope`.
