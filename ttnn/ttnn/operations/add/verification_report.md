# Verification Report: add

Target: Quasar (`quasar`, 8×4 = 32 clusters) on the ttsim simulator. Specification: `eval/specs/add/spec.py`
(lint: PASSED, 0 errors, 6 unresolved items that are the specification's open questions, not defects).
Golden results: `/tmp/add_golden_verify/` (copied: `verifier_report.json` in this directory).

## Code Review

The Phase 0 build is close to the design and needed only small fixes.

Fixed:
- **Dead constant / duplicated literal (DRY)**: `add_program_artifacts.py` declared `NEOS = 4` (unused), while
  `validate_thread_plan` restated the valid compute-thread counts as a literal `(1, 2, 4)` in both the check and
  its message. Replaced both with one `COMPUTE_THREAD_COUNTS = (1, 2, 4)` constant.
- **L1 ledger currency** (see below): two stale symbol-table rows corrected.

Checked, no change needed:
- **Specification rules**: R1 (float32 + `fp32_dest_acc_en=False` refused in `validate()` before SUPPORTED), R2
  (`default_compute_kernel_config()` defined once, exported, the only source of the `None` default; lint
  `default_factory` passes), R3 (`validate()` reads neither `math_fidelity` nor `math_approx_mode`; both are passed
  through to `ComputeHardwareConfig`), R5/R9 (DRAM and L1 cells pass; output allocated from `a.spec`, so DRAM→DRAM
  and L1→L1; asserted in `test_add_plan.py`), R6 (every out-of-SUPPORTED cell refused with `NotImplementedError`
  subclasses: 3046 xfail_expected, 0 wrong mode), R10 (`memory_axes()` applied to **each** tensor operand; a sharded
  operand is refused by name). R4, R7, R8 are not in reach yet (their cells are refused); they re-arm with
  Refinements 1 and 4. Rejection X1 raises `ValueError` before any support check (passes).
- **One dispatch**: exactly one `ttnn.generic_op` per call; no host-side ttnn op.
- **Host program**: Metal 2.0 `ProgramSpec` only (no `ProgramDescriptor`), as Quasar requires.
- **Kernels**: `void kernel_main()`, `api/...` include paths, `TensorAccessor` for every tensor, no deprecated
  address generators. Compute uses the eltwise chain helper `compute_kernel_lib::add` with `InputTileMapping::Block`
  set explicitly on both inputs (the `Scalar` default would silently read one tile per block), `PerBlockSize`
  wait/pop/reserve/push, `static_assert(block_tiles <= DEST_AUTO_LIMIT)`, `fpu_math_fidelity` always set (ttsim gap).
- **DFB sync**: reader posts one credit per page through implicit sync (ISR), compute waits/pops `block_extent` per
  block, Σ = the NEO's tiles; compute pushes per block, writer drains one entry per tagged write. Push = wait on all
  three rings, including ragged tails (tested at block 1/2/4 under C = 1 and C = 4).
- **Tag hand-back**: reader and writer end with `noc_async_{read,write}_set_trid(0, …)`; the session test (a tt-metal
  program before, `add` twice, `to_layout(ROW_MAJOR)` after, one device session, L1 operands) passes.
- **Broadcast efficiency**: no broadcast in Phase 0; no filled/replicated tiles.
- **Helpers**: the DM side has no helper for tensor-page ↔ ring transfers (design, *Helpers considered and
  rejected*); raw tagged `noc.async_read/async_write` is correct. No multicast/semaphores exist.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.add.spec`; no op-local axis function.
- `SUPPORTED` equals `PHASE0` axis for axis (lint confirms) and names every axis the kernel gates on, including
  `memory_config`, `sharding`, `fp32_dest_acc_en`, `output_dtype`.
- `EXCLUSIONS = []`. `validate()` order: X1 shape check → axes → R1 → SUPPORTED per axis → b's own
  dtype/layout/placement against SUPPORTED and against a's → EXCLUSIONS. Refusals are
  `UnsupportedAxisValue` / `ExcludedCell` (`NotImplementedError` subclasses).
- `add()` calls `validate()` as its first statement.
- The op file declares no `INVALID`.
- No drift fixes needed (xpass_drift = 0).

**INVALID audit (specification)**: one entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}` — single tensor (both
axes describe the operands), structural (block floats have no row-major form), the canonical bf8b+RM entry is
present. No cross-tensor coupling; no "not yet" entries (float32 + 16-bit DEST is correctly a RULE, R1, not
INVALID). No weight axes, so no canonicalization cells apply. Nothing to report.

## Design Conformance

- **Algorithm**: FPU add of bf16 tiles into fp32 DEST, rounded once at pack: as designed.
- **Pipeline topology**: reader (a, b per page, implicit sync) → compute (one chain call per NEO) → writer
  (implicit sync): as designed.
- **Work distribution**: flat page ranges over the full grid via `split_work_to_cores(..., row_wise=True)` in rounds
  of C pages; clusters engaged = `min(num_tiles, 32)` (tested at 13 / 52 / 263 / 1050 tiles, imbalance ≤ 1 tile).
  Both DM halves are batched: implicit sync keeps up to a ring's free entries in flight with no per-page barrier on
  either side. The grid is filled.
- **Thread plan**: read off the host program: `THREAD_PLAN = (1, 1, 1)`, one named constant, passed as CT args
  `reader_threads` / `compute_threads` / `writer_threads` to all three kernels, which size their loops from them and
  `get_my_thread_id()`. The design states the op as **data-movement bound** (one FPU op per tile, 2 reads + 1 write
  per tile) with the reason, matching the target page's *Choosing the plan* and the twin op's measured floor
  (`toy_thread_plan`: 84,562 / 84,542 / 85,938 cycles at (1,1,1) / (2,2,2) / (4,4,2)). Rings are sized per NEO
  (`× C`). Tests show every thread ran (plans (1,1,1), (2,2,2), (4,4,2), (1,4,1) into a NaN-prefilled output) and the
  session ordering test passes. No `pipeline_logs/model_cycles.jsonl` exists: this backend has no timing model, so no
  plan comparison is measured here. Conformant.
- **Blocking-model fidelity**: `BLOCK_TILES`, `RING_DEPTH_BLOCKS`, `MIN_TILES_PER_CORE`, `THREAD_PLAN` are each
  defined once; ring capacities (`ring_depth × block × C`), the compute CT arg, the clamp and the split derive from
  them; the builder takes overrides for each (tested). No CB scales with an op dimension. The per-core compute loop
  works in blocks of `block_tiles` (one DEST section), not single tiles; reader and writer issue per page with no
  per-page completion boundary (implicit sync), so the schedule reads as blocks on all three kernels.
- **Mirrored hardware constant (advisory)**: `dest_section_tiles()` in Python mirrors `get_dest_limit()` of
  `dest_helpers.hpp` (no Python query exists). Drift is caught by the kernel's `static_assert`; acceptable.
- **Advisory**: the design's Parameters table and Blocking Model state the block clamp as
  `min(BLOCK_TILES, ceil(max_core_tiles / C))`; the implementation also clamps to the config's DEST section
  (recorded by the implementer as an advisory deviation). Harmless and safer; the ledger now says so.

## L1 Ledger Audit

- **Currency**: three DFB rows (`cb_input_a_tiles`, `cb_input_b_tiles`, `cb_output_tiles`) match the three
  `DataflowBufferSpec`s; capacity `RING_DEPTH_BLOCKS × block_tiles × C` matches `entries = ring_depth * block * C`.
  Fixed in place: the `block_tiles` bound now names the DEST-section clamp the host applies; the
  `RING_DEPTH_BLOCKS` predicate now names the real checks (`ring_depth >= 2`, upper bound via the per-ring and
  footprint asserts) instead of "host constant".
- **Capacity vs live set**: each ring's capacity is two blocks per NEO for a one-block live set; the gap is the stated
  double buffering (overlap of read/add/write). No under-sized spanned axis: the only spanned axis is `tiles`
  (`block_tiles`), and capacity scales with it.
- **Page format vs DEST width**: inputs are dataflow-only bf16 payloads (unpacked to srcA/srcB, not to DEST); the
  output ring is bf16 under fp32 DEST, justified by the output ABI (R8: the sum rounded once into the output dtype)
  with no later phase reading it. No finding.
- **Disjoint lifetimes**: none — all three rings live for the whole program and are concurrently live; every
  "Shares with / why not" cell is filled.
- **Bounds / closed form**: footprint = `3 × RING_DEPTH_BLOCKS × block_tiles × C × tile_bytes` (49,152 B in Phase
  0, 1.2% of `get_max_worker_l1_unreserved_size()`), asserted per ring (≤ 1,048,560 B) and in total against the
  queried budget. No op dimension appears.
- **Data-movement budget**: present and consistent with the code: a, b, out each cross their tier once; no
  cross-core traffic; cheapest-traffic split implemented (all candidate splits tie on bytes; occupancy chose the flat
  split). One finding that is not about the split: tile-padded pages of the YOLOv8 H = 2 shapes carry 14 padding
  rows per tile, half of them in faces that are entirely padding. Folded into perf Refinement 3, not a standalone item.
- **Block-size defaults**: interleaved default (spread across the full grid, then the coarsest block) departs to
  `block_tiles = 4` (one DEST section) on the design's stated mechanism cap and overlap argument, backed by the twin
  op's measurement (blocks of 1/4/8 within 2% at the floor). Accepted.

## Precision Baseline

`tests/ttnn/unit_tests/operations/add/test_add_precision_baseline.py` — bfloat16 operands drawn standard normal,
TILE, DRAM, default config (fp32 DEST, HiFi4); reference = float64 sum of the stored inputs.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP (bf16) | Elements ≠ 0 ulp | got/true ratio median [p5, p95] |
|-------|-----|-------------|--------------|------------------|----------------|------------------|-------------------------------|
| (1,1,17,50) | 0.9999984 | 0.0078 | 0.00147 | 0.00177 | 0 | 0 | 1.0000 [0.9972, 1.0029] |
| (1,2,8400) | 0.9999984 | 0.0156 | 0.00147 | 0.00178 | 0 | 0 | 1.0000 [0.9972, 1.0028] |
| (1,1,1600,128) | 0.9999984 | 0.0156 | 0.00148 | 0.00180 | 1 | 9.8e-6 | 1.0000 [0.9972, 1.0028] |
| (2,4,128,512) | 0.9999984 | 0.0156 | 0.00147 | 0.00180 | 1 | 7.6e-6 | 1.0000 [0.9972, 1.0028] |

**Assessment**: the result is the correctly rounded bf16 sum: the errors are the single output rounding (half a
bf16 ulp, relative ≈ 2^-9), and the ratio spread is symmetric about exactly 1.0 — no scale or structural error. The
rare 1-ulp elements (≈1e-5 of elements) are within the specification's 1-ulp bfloat16 grade (NUMERIC_RESOLVED
`('bfloat16', 'default')`): most likely operand pairs whose exponents differ enough that the fp32 DEST sum itself
rounds, or packer tie-breaking. The golden suite's six bf16 numeric regression cases (small/large magnitude,
uniform) also pass.
**Recommended tolerances**: bfloat16 cells are graded by the numeric list (1 ulp), which holds; defaults hold for
the PCC/RMS gate. No op-specific limits needed.

## Verifier CLI Summary

- supported_pass: 86 (80 PHASE0 cells + 6 bf16 numeric regression cases)
- xfail_expected: 3046
- invalid_skipped: 1560 (780 INVALID bf8b×ROW_MAJOR + 780 bf8b×TILE unbuildable on quasar)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 2 (rejection X1, passed; the empty LOOSE_CASES placeholder, skipped)
- queue_gaps: 0 (`--queue op_requirements.md`)

Golden runner: 87 passed / 4694 total, 0 failed, 0 errors, 0 hangs. Unit tests: 33 passed (7 acceptance, 22 plan,
4 precision baseline).

## Recommendations

- **Queue** (`op_requirements.md`): R1 broadcasting (all five values; scheme-change, first), R2 ROW_MAJOR (native
  tilize/untilize), R3 perf (half-page transfers on H ≤ 16 tile-rows: the YOLOv8 dist2bbox shapes), R4 numeric
  bundle (float32 via the exact SFPU path, 16-bit DEST, output dtype), R5 perf (per-regime thread plan for the
  compute-heavier RM and float32 paths). `bfloat8_b` is omitted on purpose (unbuildable on quasar).
- No LOOSE_CASE is perf-flagged, so the perf regions are free-selected; on ttsim both perf refinements are gated on
  structural counts and their time gain is unconfirmed until a timed backend measures them.
- The R1 resident b ring is the first op-dimension-dependent buffer: it needs a bound and a streaming fallback in
  the ledger before it lands.
- Not served by the specification (DECISIONS): the trace's height-sharded call is refused by name (R10); a
  sharded TARGET value would be a local (per-core) shard here, every axis being independent.
- Edge case outside TARGET: a zero-volume operand would make `split_pages` produce no cores (`max()` of an empty
  dict). No INPUTS entry has one; worth a guard if the specification ever admits it.
- Hardware confirmation of the simulator's numerics (the 1-ulp elements, the twin op's cycle figures) is a finding
  for a hardware run, not a refinement.
