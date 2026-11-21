# Verification Report: nlp_concat_heads

Target quasar (8×4 clusters, 2 DRAM banks per `device.dram_grid_size()`), backend ttsim (untimed: no timing
below is a performance result). Specification `eval/specs/nlp_concat_heads/spec.py`; lint
(`python3 -m eval.validate_contract nlp_concat_heads --phase0`) PASSED, 0 errors, 5 UNRESOLVED open items
(the specification's own: `governs`, `special_values`, `freedoms`, `PERFORMANCE.bar/measure`), none of which binds Phase 0.

## Code Review

The implementation is a faithful build of the design's `tile_relay` regime and needed little. Fixed:

- **Regime guard in the program builder** (`nlp_concat_heads_program_artifacts.py::head_geometry`). It used
  `head_dim // 32`, which silently floors a head dim that is not a multiple of 32. Only `validate()` stood
  between a w_non_aligned call and a wrong whole-tile walk. It now raises `NotImplementedError` naming the
  unbuilt `subtile_assemble` regime, so a `validate()` widened ahead of that regime (the risk the design's
  "Validate wider than PHASE0" row names) cannot dispatch wrong bytes. Pinned by the new plan test
  `test_tile_relay_refuses_a_head_dim_off_the_tile_grid`.
- **L1 ledger live set** (`l1_ledger.md`): it stated `RING_DEPTH_TILES` pages, which is true only at plan (1, 0, 1).
  Corrected to `RING_DEPTH_TILES` per DM thread endpoint × `max(R, W)`, matching the capacity expression
  and `DataflowBuffer.md:255`. Fixed a stale audit-number reference.
- **Precision baseline test added** (`test_nlp_concat_heads_precision_baseline.py`), see below.

Checked and found conformant (no change):

- **Specification rules.** R1: no compute kernel, DM moves raw pages; float32 bit-exact through the program
  (baseline). R2: output shape / dtype / TILE. R3: `_output_memory_config` (None → input buffer, interleaved);
  a probe confirmed the L1 and None placements. R4: `canonical_memory` from the spec reads the config by
  content, and a non-config raises `TypeError` (X2). R5: an L1 input is read by the same accessor (probe). R6:
  input axes from `eval.memory.memory_axes`, so a sharded input / output config is refused by name. R7 / X3: row-major
  is refused through `SUPPORTED["layout"]`. R8: everything outside SUPPORTED raises a `NotImplementedError`
  subclass (206 xfail_expected, 0 wrong mode). X1: rank ≠ 4 raises `ValueError` before anything else.
  GUIDANCE followed (whole-tile walk in output order, no conversion, split finer than a token row).
- **Kernels.** `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor` (no `InterleavedAddrGen`).
  Implicit sync on both ring ends: one tagged transfer per entry, no reserve / barrier / push / wait / pop, so
  push count equals wait count by construction. The sticky trid is handed back as the last statement of both kernels
  (target page, ring protocols). The reader seeds its (c, h, r, b) counters by division once and steps by
  increment, with no per-tile division.
- **Helpers.** No compute phase, so no compute helper applies. No dataflow helper wraps an implicit-sync page →
  ring transfer (design §Block Operation Realization). No multicast / semaphore code exists to migrate.
- **Broadcast efficiency.** N/A: one operand, and each tile is read by exactly one core.

## Design Conformance

- **Algorithm / topology**: reader → `relay` ring → writer, no compute: matches. One `ttnn.generic_op` of one
  `ttnn.ProgramSpec` (Metal 2.0; no `ProgramDescriptor`), so the one-dispatch rule holds.
- **Work distribution / fills the machine**: flat output-page split over the whole queried grid,
  `split_work_to_cores(..., row_wise=True)`, contiguous range per cluster (± 1 tile). Engaged clusters =
  min(32, tiles), pinned by `test_engaged_cores`. Both NoC halves stream: reads are bounded only by free ring
  entries (8 in flight); writes leave per entry with no barrier. No serialized half.
- **Thread plan**: read off the host: `THREAD_PLAN = (1, 0, 1)`, one named constant, passed as compile-time
  args `reader_threads` / `writer_threads`. Each kernel strides by its count from `get_my_thread_id()`. The ring
  is sized `RING_DEPTH_TILES × max(R, W)`. The design states the op as **bound by data movement**, with
  reason (no compute per tile), so one
  hart per direction is the sanctioned plan, and C = 0 because there is nothing to compute (the target page's
  (1, 1, 1) has a NEO only because those ops compute). Required tests present and passing: every hart ran
  (`test_knobs_give_the_same_bytes` at (2,0,2), (2,0,1), (1,0,2), (4,0,2) on 37- and 300-tile shapes whose counts
  do not divide by the thread count); one device session after another program (a tt-metal tilize and the op
  itself) and before `ttnn.to_layout(..., ROW_MAJOR)` (`test_one_device_session_with_tt_metal_programs`). No
  timing model on this backend, so the plan alternatives were not timed (no `model_cycles.jsonl` exists for the phase).
- **Blocking-model fidelity**: every knob is a parameter defined once (`THREAD_PLAN`, `RING_DEPTH_TILES`,
  `BLOCK_TILES`, `MIN_TILES_PER_CORE` in `nlp_concat_heads_program_artifacts.py`). Every dependent quantity is
  derived from it (ring entries, per-core `block_tiles` RT arg, engaged clusters), and each is exercised at non-default
  values. No CB scales with a tensor dimension. Expression: the block loop sets only the enclosing trip count.
  That is not an expression failure, because no unit is independently synchronized (implicit sync, no per-tile
  completion boundary), and the design records that per-tile cost does not amortize with block size here. No
  half-turned split: the per-core loop walks the whole range as one block (`block_tiles = core_tiles`).

## Registry Conformance

- `INPUT_TAGGERS = AXES` imported from the specification; no op-local axis function.
- `SUPPORTED` declares all seven axes and equals `PHASE0` exactly (lint-checked).
- `EXCLUSIONS = []`. `validate()` checks SUPPORTED per axis, then EXCLUSIONS, raising
  `UnsupportedAxisValue` / `ExcludedCell`; the entry point calls it first.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix was applied: xpass_drift is 0. (A probe shows every knob-turn axis already works through
  the program, but Phase 0's SUPPORTED must equal PHASE0. Widening is Refinement 2.)
- **INVALID audit (specification)**: `INVALID = []`, correct for this rectangle. TARGET has no bf8b dtype
  (no block-float on quasar), and layout is TILE-only with row-major refused by rejection X3, so the canonical
  bf8b+ROW_MAJOR entry has nothing to cover. No weight axes, so no canonicalization cells. No cross-tensor
  coupling. No specification defect found.

## L1 Ledger Audit

- **Currency**: one ring, `relay`, declared as `DataflowBufferSpec(entry_size=input.buffer_page_size(),
  num_entries=RING_DEPTH_TILES × max(R, W))`. That matches the ledger row and symbol table (live-set wording fixed, see
  Code Review).
- **Capacity vs live set**: capacity = live set (depth is the in-flight count). The axis accounting streams over
  every axis and spans none, which is correct for a FIFO relay. Neither over- nor under-sized.
- **Page format vs DEST**: no DEST in the path (dataflow-only payload); float32 pages are the output ABI. No finding.
- **Disjoint lifetimes**: one buffer; the ledger justifies why it cannot be removed (a NoC read lands in L1).
- **Bounds / closed form**: total `RING_DEPTH_TILES × max(R, W) × page_bytes` = 16 KiB bf16 / 32 KiB fp32 at
  Phase 0. All symbols are bounded, and the host asserts the ring cap and the unreserved-L1 query. The footprint is
  constant in B, H, S, D and `block_tiles`.
- **Data-movement budget**: each tensor crosses DRAM once, with no cross-core traffic. It is consistent with the
  implemented split. The cheapest-traffic split is the implemented one (all candidates tie on bytes; occupancy decides).
- **Block-size defaults**: interleaved, so work is spread over the full grid and each core takes its whole range
  (the coarsest block). Held.

## Precision Baseline

`tests/ttnn/unit_tests/operations/nlp_concat_heads/test_nlp_concat_heads_precision_baseline.py` (seeded normal inputs).

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median [p5, p95], std |
|-------|-------|-----|-------------|--------------|------------------|---------|--------------------------------------|
| (1,4,32,64) | bf16 (entry point) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0], 0 |
| (1,4,32,128) | bf16 (entry point) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0], 0 |
| (1,32,128,64) | bf16 (entry point) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0], 0 |
| (1,64,32,128) | bf16 (entry point) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0], 0 |
| (1,4,32,64) | fp32 (program, validate bypassed) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0], 0 |
| (1,4,32,128) | fp32 (program) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0], 0 |
| (1,64,32,128) | fp32 (program) | 1.0 | 0 | 0 | 0 | 0 | 1.0 [1.0, 1.0], 0 |

**Assessment**: bit-exact, as R1 requires of a movement op. No scale or structural signature: the ratio is
exactly 1 everywhere. float32 survives the relay unchanged (no DEST, no format conversion).
**Recommended tolerances**: exact (every cell resolves to `exact` in `NUMERIC_RESOLVED`).

## Verifier CLI Summary

Golden run `eval/eval_test_runner.sh eval/golden_tests/nlp_concat_heads/` (8 sim workers, forked), then
`python3 -m eval.verify_supported` (report copied to `verifier_report.json` in this directory):

- supported_pass: 10
- xfail_expected: 206
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 4 (rejections X1, X2, X3 — all passed — and the empty `test_op_loose` parameter set, skipped)
- Queue coverage (`--queue op_requirements.md`): 7 values of TARGET − SUPPORTED, 0 gaps.

Local tests: acceptance 4/4, plan 34/34 (one new), precision 7/7.

## Recommendations

- **Queue shape.** The 206 xfail cells reduce to one scheme-change (`w_non_aligned` → `subtile_assemble`,
  Refinement 1) and one validate-only bundle (Refinement 2, all knob-turns). A verifier probe that bypassed
  `validate()` ran float32, L1 input, L1 and None outputs, h_non_aligned and batched through the current program,
  bit-exact and placed per R3. That gives two generality refinements, not 3–5: splitting the bundle would be
  artificial fragmentation. Perf slots follow on the 2:1 cadence (Refinement 3, bank stagger), then Refinement 4
  (assembler harts) once generality is exhausted.
- **Perf lamps with no structural signal on this backend (not queued).** (a) Occupancy: the traced calls
  give each of 8 / 16 clusters one tile, so launch and arg setup may dominate. `MIN_TILES_PER_CORE` is the knob
  (exposed and tested), but which direction wins cannot be stated as a structural count. Measure 1 / 4 / 8 on
  `ttsim-perf` or hardware. (b) Ring depth 4 / 8 / 16 and thread plan (2, 0, 2):
  check on a timed backend.
- **Numerics.** Nothing to tune: exact by construction. `NUMERIC["special_values"]` is UNRESOLVED in the specification;
  NaN / inf would also move unchanged (raw page copy), but no cell passes one.
- **L1.** Footprint is shape-independent (≤ 32 KiB at fp32 for the current plan); no OOM risk at any shape,
  including the models' `[1, 32, 4096, 128]` (16,384 tiles, 512 per cluster).
- **Hardware-only checks** (findings, not refinements): every perf gate here is structural; the time gain of
  Refinements 3 and 4 needs a timed backend or silicon.
