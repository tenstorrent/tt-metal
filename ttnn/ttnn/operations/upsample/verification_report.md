# Verification Report: upsample

Target: Quasar (8×4 = 32 clusters) on the ttsim simulator (no clock, no timing model). Specification:
`eval/specs/upsample/spec.py` (lint: PASSED, 0 errors, 5 UNRESOLVED open items that are the
specification's own, none blocking). Golden results: `/tmp/upsample_golden/` (report copied to
`ttnn/ttnn/operations/upsample/verifier_report.json`).

## Code Review

Fixed / added in this pass:
- **R6 test was missing.** The spec's review check for R6 says "a sharded input is a test the op carries
  itself"; there was none. Added `test_upsample_placement.py::test_height_sharded_input_refused_by_name`
  (the trace's sharded call, `[1,4,4,64]` height-sharded in L1, shard `[8,64]`): refused by
  `validate()` with `UnsupportedAxisValue` naming `sharding='height'`. Passes.
- **R5, DRAM half**: added `test_dram_input_gives_dram_interleaved_output` (the L1 half already existed in
  `test_upsample_threads.py`). Passes.
- **Precision baseline** test added (below).

Reviewed and found conformant (no change needed):
- **Specification rules.** R1: byte mover, no compute kernel, ring `data_format_metadata` = tensor dtype;
  float32 bit-exact on the RM path (verifier probe `probes/probe_002.py`). R2: `normalize_scale_factor`
  maps `2` and `(2, 2)` to the same `(2, 2)`; the form is never gated (golden cells with int and pair
  factors both pass). R3: DRAM and L1 cells pass. R4: every out-of-SUPPORTED cell xfails with
  `NotImplementedError` (142/142). R5: output `TensorSpec` built from `input.memory_config()`. R6:
  `memory_axes()` drives the gate. REJECTIONS X1–X3: raise `ValueError` before the support gate (golden
  `test_rejection[X1..X3]` pass).
- **Kernels.** `void kernel_main()`; `api/dataflow/...` includes; `TensorAccessor` through tensor
  bindings; push count = pop count (1 entry per block on both sides, ragged last block keeps 1 entry);
  every NoC transfer tagged `trid 0`, barrier on the same id before `push_back` / `pop_front`; tag handed
  back as the last statement (`noc_async_{read,write}_set_trid(0, …)`), and `test_session_order`
  (tt-metal `to_layout` → upsample ×2 → tt-metal `to_layout` ×2, one device session) passes.
- **Helpers.** No kernel-library helper expresses "interleaved RM page → DFB entry at a per-stick offset"
  or "one L1 stick → several interleaved pages" (design's rejected-helpers table checked: compute helpers
  are tile/compute-side, `local_copy_helpers_dataflow` targets the core's own L1 with fixed coordinates,
  mcast has no reuse-shared operand). Raw Device-2.0 NoC/DFB API is the sanctioned realization.
- **`kernels/upsample_compute.cpp`** is a comment-only file not bound to the program. It is kept because
  the pipeline lists it as a required output (implementer breadcrumb); it documents why the op has no
  compute stage. Not a defect.

## Registry Conformance

- `INPUT_TAGGERS = AXES` imported from the specification; no op-local axis function.
- `SUPPORTED` lists all 7 axes (dtype, layout, memory_config, sharding, mode, channel_alignment, scale)
  and equals `PHASE0` (lint-checked).
- `EXCLUSIONS = []`.
- `validate()`: malformed-call errors (X1–X3, `ValueError`) → axis derivation (`memory_axes` + taggers) →
  SUPPORTED per axis (`UnsupportedAxisValue`) → EXCLUSIONS (`ExcludedCell`) → the `stick_stream` regime
  predicate (`NotImplementedError`, deferred `channel_chunk`). The public entry calls it first.
- The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED (no XPASS drift). The probe shows float32 / non-aligned C / per-axis
  already work on the RM path with the gate bypassed, but SUPPORTED must equal PHASE0 in the first phase
  (lint), and those cells were refused, not XPASS: they are Refinement 2.
- **INVALID audit (specification).** One entry, `{dtype: bfloat8_b, layout: ROW_MAJOR}`: single tensor,
  structural (bf8b is block-quantized, ROW_MAJOR has no blocks), the canonical activation entry. No
  weights, no cross-tensor coupling, no canonicalization cells needed. Well-formed.

## Design Conformance

- **Algorithm / pipeline**: as designed. Reader → STRIDED DM→DM ring → writer, no compute; each input
  stick read once, written `sh·sw` times from the one resident copy.
- **Work distribution**: `split_work_to_cores(grid, N·H·W, row_wise=True)` over the queried grid
  (all 32 clusters for every INPUTS entry but `[1,4,4,8192]`, 16 sticks → 16 clusters, the design's
  stated `channel_chunk` deferral). Both dataflow halves batched: one barrier per block on each side.
- **Thread plan**: read off the host program. One constant `THREAD_PLAN = (2, 4)` → reader
  `num_threads = 2`, writer `num_threads = 4`, C = 0; the counts reach the kernels as CT args
  (`reader_threads`, `writer_threads`) and loops stride by them with `get_my_thread_id()`. Ring capacity
  `RING_ENTRIES_PER_THREAD · max(R, W)` = 8 (per-slice accounting). Both kernels list the ring out of
  implicit sync and tag every transfer, as the page requires for explicit sync. Tests showing every
  thread ran (`test_each_{reader,writer}_thread_*`, mask probes against the predicted block owners)
  and the session-order test are present and pass. The design states the op as **bound by data
  movement** and gives the reason it still uses more than one thread of each kind (explicit sync forced
  on the writer: one entry feeds `sh·sw` writes).
  **Not measured against the alternatives** `(1,0,1)`, `(1,0,2)`, `(2,0,2)`, `(1,0,4)`: this backend
  has no timing model (backend page), and `pipeline_logs/model_cycles.jsonl` does not exist for this run.
  That comparison needs a `ttsim-perf` run — a finding, not a refinement.
- **Blocking-model fidelity**: `block_sticks`, `RING_ENTRIES_PER_THREAD`, `BLOCKS_PER_WRITER_THREAD`,
  `THREAD_PLAN`, `L1_RESERVE_BYTES` are host parameters, each defined once; `block_sticks` is computed
  once and feeds the ring's `entry_size` and both kernels' CT arg; `stick_stride` likewise. No CB scales
  with a whole-op dimension (the ring is `block_sticks · stick_stride · 8`, clamped to the ring cap).
  Expression: reader = 1 reserve / `extent` reads / 1 barrier / 1 push per block; writer = 1 wait /
  `extent·sh·sw` writes / 1 barrier / 1 pop per block — no per-unit handshakes on either side.
- **Block-size default (departure noted)**: the interleaved default is "spread over the grid, then the
  coarsest block that fits"; here the per-cluster range is cut into `W · BLOCKS_PER_WRITER_THREAD = 8`
  blocks so that every writer thread gets work and blocks overlap. The `≥ W` part is structural (W
  threads need ≥ W entries); the factor 2 is a reasoned, unmeasured choice (design Perf lamp
  "Overlap"). Its side effect on the model shape — a ragged cut 7×7 + 1 giving writer threads
  14/14/14/8 sticks — is Refinement 3's gate.

## L1 Ledger Audit

- **Currency**: one CB (`cb_input_sticks`), the only `DataflowBufferSpec` in the program; size
  expression `RING_ENTRIES · block_sticks · stick_stride` matches `create_program_artifacts`.
- **Capacity vs live set**: 8 entries = 2 per writer slice (double buffering); with ≤ 8 blocks per cluster
  in Phase 0 a cluster's share can be fully resident. Axis accounting is stated (`stick` spans
  `block_sticks`, `channel` spans whole stick, `rep_h`/`rep_w` never stored) and true. No over/under.
- **Page format vs DEST**: dataflow-only payload, no DEST, no compute config — audit not applicable,
  and the ledger says why.
- **Disjoint lifetimes**: single buffer; n/a.
- **Bounds and closed form**: every symbol bounded; `stick_bytes` bounded by the regime predicate;
  total ≤ `ring_bytes_cap = min(1,048,560, unreserved − reserve)`. Phase 0 max 131,072 B
  (`[1,4,4,8192]`); 57,344 B for `[1,40,40,512]`.
- **Data-movement budget**: present and consistent with the code (input 1 crossing, output 1 crossing,
  0 cross-core bytes). The cheapest-traffic split is the one implemented; the tie (`channel_chunk`) is a
  deferred row with a positive reason (occupancy-only, one INPUTS entry).
- Per-core footprint: `RING_ENTRIES_PER_THREAD · max(R, W) · block_sticks · stick_stride`; scales with
  depth knob, plan, block (inversely with `BLOCKS_PER_WRITER_THREAD`), stick width / dtype; independent
  of the factors.
- Nothing to fix; no findings to fold into a refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/upsample/test_upsample_precision_baseline.py` (bf16 RM; reference in
float64 per the spec):

| Shape / factor / buffer | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true ratio median [p5, p95] | Bit-exact |
|---|---|---|---|---|---|---|
| (1,16,16,32) ×2, DRAM | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] | yes |
| (2,8,8,128) ×3, L1 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] | yes |
| (1,20,20,512) ×(2,2), DRAM | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] | yes |
| (1,40,40,512) ×(2,2), DRAM | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 [1.0, 1.0] | yes |

ULP error: 0 everywhere (bit-identical). Verifier probe: float32 (4 shapes), non-aligned C (3 shapes),
per-axis factors (4 shapes) on the RM path with the gate bypassed are also bit-exact.

**Assessment**: exact, as R1 and `NUMERIC_RESOLVED` (class `movement`, metric `exact`) require. No scale
or structural signature (ratio exactly 1.0).
**Recommended tolerances**: exact (every cell resolves to `exact` in `NUMERIC_RESOLVED`).

## Verifier CLI Summary

- supported_pass: 26
- xfail_expected: 142 (float32, TILE, non_tile_aligned, per_axis cells — Refinements 1 and 2)
- invalid_skipped: 84 (42 bf8b+ROW_MAJOR INVALID, 42 bf8b+TILE unbuildable on Quasar)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 4 (3 rejection tests, passing; 1 empty `test_op_loose[NOTSET]`, no LOOSE_CASES)
- Queue coverage (`--queue`): 5 TARGET−SUPPORTED values, 0 gaps (bfloat8_b recorded as `Omitted`).

Unit tests: 22 pass (acceptance 7, thread plan / session / regime / R2 / R5-L1 9, placement 2,
precision 4).

## Recommendations

- **Queue**: R1 TILE layout (scheme-change, hardest, first) → R2 float32 + non-aligned C + per-axis on
  both layouts (knob-turns; gate-only on RM per the probe) → R3 perf on the YOLOv8 RM calls (structural
  gate) → R4 perf on the TILE path. No LOOSE_CASE is perf-flagged, so the perf target is free-selected:
  the model's three calls, all already supported.
- **Thread-plan measurement** (finding, needs another backend): the (2, 0, 4) plan against `(1,0,1)`,
  `(1,0,2)`, `(2,0,2)`, `(1,0,4)` can only be ranked on `ttsim-perf` or hardware. The target page's default
  for a data-movement-bound op is (1,1,1)-like; whether 6 harts beat 2 is open.
- **Compile-time shape args**: `in_w`, `scale_h`, `scale_w`, `block_sticks`, `stick_bytes` are CT args, so
  each distinct shape/factor compiles a new kernel pair. Fine for the model's three calls; if a caller
  sweeps many shapes, moving `in_w` and `block_sticks` to runtime args trades a little loop folding for
  program-cache hits. Not a refinement (no failing cell, no structural count).
- **Small sticks** (`C = 32` bf16, 64 B pages: `[1,64,64,32]`, `[1,256,4,32]`): one NoC transaction per
  page is fixed by the interleaved RM layout; no block size changes the transaction count. Only
  `trid_double_issue`-style in-flight depth helps; part of R3's lever set, not its target.
- **Occupancy on `[1,4,4,8192]`** (16 of 32 clusters, 1 stick each): the deferred `channel_chunk` regime
  would fill the grid; the gain is bounded to this one INPUTS entry (1 MiB of output). Left deferred, as
  the design states.
- **Specification open items** (not defects): `NUMERIC.special_values` (NaN/Inf moved unchanged — true
  by construction for a byte mover), `PERFORMANCE.bar/measure` UNRESOLVED.
