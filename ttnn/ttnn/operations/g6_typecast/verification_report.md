# Verification Report: g6_typecast

Phase 0 state: regime R1 `tile_stream` (interleaved TILE → compute typecast → interleaved TILE), contiguous
tile ranges over the full compute grid, SUPPORTED = `bf16 → {float32, bfloat8_b, int32}`, TILE, DRAM/L1,
all three alignments, rank 3/4.

## Code Review

### Fixed
- **Program-cache key omitted the block knobs (BB-07).** `_cache_key` keyed on shape/dtype/layout/memory but
  not on `BUFFER_DEPTH`, `BLOCK_TILES_CAP`, `L1_OP_BUDGET_BYTES`, `L1_SLACK_BYTES`, all of which change CT args
  and CB sizes. Turning a knob (as a perf refinement will) would have replayed a stale program; the blocking
  test only worked because it cleared `_PROGRAM_CACHE` by hand. The four knobs are now part of the key
  (`g6_typecast.py`, reading them from the descriptor module — single source).

### Reviewed, no change needed
- **Registry wiring**: INPUT_TAGGERS (all `(inputs, axes)`), SUPPORTED, EXCLUSIONS (empty, as the prompt rule
  requires), `validate()` (SUPPORTED per-axis → EXCLUSIONS, support refusals from `_op_contract`) is the entry
  point's first line. No `INVALID` in the op file.
- **Prompt rules** (`eval/prompts/g6_typecast.txt ## Rules`) that apply to Phase 0: conversion rules (bf16→fp32
  exact, bf16→int32 truncation, bf8b RNE-precise packing) — met, bit-exact on the golden + baseline; 32-bit
  destinations use 32-bit DEST — met; EXCLUSIONS empty, no size axis — met; output keeps the input's layout
  and memory config — met (allocated from `input_tensor.memory_config()`); one program, no ttnn conversion
  op called — met; core count is the blocking decision (`split_work_to_cores` over the full grid) — met;
  BB-03 final write barrier — met; BB-08 `last_plan` — met. ROW_MAJOR / 32-bit-source rules re-arm with R1/R2.
- **Blocking-model fidelity**: `block_tiles` is one host value (from `plan_blocks`) passed as one CT arg to
  reader, compute and writer; both CB page counts derive from `effective_depth · block_tiles`; `BUFFER_DEPTH`
  and `BLOCK_TILES_CAP` are tunable knobs (exercised by `test_g6_typecast_blocking.py` at caps 1/3/4/7). No
  CB scales with a whole-op dimension; the per-core range is bounded by the L1-clamped `plan_blocks`. DRY holds.
- **Expression**: reader issues `block_extent` reads then ONE barrier and ONE push per block; compute is one
  eltwise-chain call per block with Upfront wait / AtEnd pop and Upfront reserve / AtEnd push (one handshake
  per block per CB, DEST sub-blocking internal to the helper — legitimate); writer waits once per block,
  issues all writes, flushes, pops once, ends with one barrier. No per-unit handshake on any of the three.
- **Compute realization deviation** (documented in the kernel): the design named a single
  `typecast<PerBlockSize>(FullBlock)` call, but FullBlock blocks are capped at DEST capacity
  (`chain.inl:3346`). The implementer calls `eltwise_chain` per block with `InitReconfigOwner::Chain` on the
  first block and `Caller` afterwards, so init/reconfig is paid once per kernel. Same elements as
  `compute_kernel_lib::typecast()`. Mechanism choice — not flagged (scope boundary).
- **Ragged tail**: all three kernels sync the actual extent of the last block; with depth ≤ 2 and ring size
  `depth · block_tiles` the tail block always starts at slot 0 or `block_tiles`, so it is contiguous. Verified
  by the cap=3/7 blocking tests on Q2185 (16/15 tiles per core).
- **bf8b pad-column masking**: correct face geometry for both `w_tail < 16` and `w_tail > 16`; only active when
  `W % 16 != 0` (never in production; covered by `test_bf8b_pad_column_mask`).
- **Dataflow API**: TensorAccessor, `void kernel_main()`, `api/dataflow/dataflow_api.h` includes — correct.
- **Helpers**: compute uses the eltwise chain (CopyTile → Typecast → PackTile, formats from CB formats). No
  dataflow helper owns an interleaved page-range stream; raw accessor loops are appropriate. No mcast needed.
- **Performance conformance**: fills the grid (`split_work_to_cores` full grid, row-wise), both halves batched
  per block. The design's "Overlap" perf lamp is real: the default block is the whole per-core range ⇒ one
  block per core ⇒ read, convert, write serialize per core (effective depth 1). That is the design's chosen
  default (coarsest block that fits) and is queued as perf Refinement 4, not a defect.

## Registry Conformance
- INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() present and correctly wired; op file declares no INVALID.
- No XPASS-driven SUPPORTED changes were needed (xpass_drift = 0).
- **INVALID audit** (`feature_spec.py`): one entry `{output_dtype: bfloat8_b, layout: ROW_MAJOR}` — both axes
  describe the output tensor (the output keeps the input's layout); structurally impossible (bf8b is a tiled
  block format). Passes all three sanity rules. The "bf8b input + ROW_MAJOR" canonical entry is not needed:
  bfloat8_b is not in TARGET `dtype` (input). No weight axes. No change requested.

## L1 Ledger Audit
- **Currency**: two rows (`cb_input_tiles`, `cb_output_tiles`) match the two declared CBs; size expressions
  (`effective_depth · block_tiles · page_bytes`) match `create_program_descriptor`. The implementer's
  ValidTiles tail note matches the kernels.
- **Capacity vs live set**: capacity = live set for both (depth is clamped to the block count, so a one-block
  core never allocates an idle slot). Axis accounting: `tile` spanned and capacity scales with it; folded axes
  stream. No over/under findings.
- **Page format vs DEST**: Float32/Int32 output pages only with `fp32_dest_acc_en=True` and they are the output
  ABI; Bfp8_b with 16-bit DEST; bf16 input page never passes the packer. Clean.
- **Disjoint lifetime**: none (both CBs whole-kernel, concurrent); sharing justification present.
- **Bounds / closed form**: symbol table bounds `block_tiles`, `effective_depth`, `out_l1_bytes_per_bank`; the
  total `effective_depth · block_tiles · (in_page + out_page) + out_l1_bytes_per_bank ≤ 1347776 − 65536` is
  closed-form and host-enforced (`plan_blocks` asserts cap ≥ 1).
- **Data-movement budget**: input ×1, output ×1, no cross-core traffic — consistent with the implemented split.
  Cheapest-traffic split is the implemented one (all candidate splits move the same bytes); bank-affine is a
  same-bytes perf lamp.
- **Block-size defaults**: interleaved default held — full grid, then the coarsest block that fits (the whole
  per-core range). No measured departure yet; the overlap co-tune (smaller block × depth 2) is perf R4.
- **Per-core footprint**: `effective_depth · block_tiles · (in_page_bytes + out_page_bytes)` (+ L1 output
  share); both CB terms scale with `block_tiles · effective_depth`. Peak production: P39 ≈ 64 tiles × 3136 B
  ≈ 200 KB + ~70 KB output share.

## Precision Baseline
`tests/ttnn/unit_tests/operations/g6_typecast/test_g6_typecast_precision_baseline.py` (production shapes).

| Case | Shape | Pair | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP / steps | ratio p5 / med / p95 |
|------|-------|------|-----|-------------|--------------|-------------|-------------|----------------------|
| Q2191 | (16,1,12) | bf16→fp32 | 1.000000 | 0 | 0 | 0 | max ULP 0 | 1 / 1 / 1 |
| Q2185 | (16,4096,12) | bf16→fp32 | 1.000000 | 0 | 0 | 0 | max ULP 0 | 1 / 1 / 1 |
| Q2229 | (1,1,32,128) | bf16→int32 | 1.000000 | 0 | 0 | 0 | 0 mismatches | 1 / 1 / 1 |
| Q2197 | (1,1,31,256) | bf16→bf8b | 0.999975 | 0.03125 | 0.00601 | 0.00766 | ≤ 1.000 step | 0.948 / 1.000 / 1.047 |
| Q2187 | (1,1,3844,256) | bf16→bf8b | 0.999975 | 0.03125 | 0.00597 | 0.00763 | ≤ 1.000 step | 0.954 / 1.000 / 1.048 |
| P39 | (1,128,256,256) | bf16→bf8b | 0.999975 | 0.03125 | 0.00597 | 0.00764 | ≤ 1.000 step | 0.955 / 1.000 / 1.048 |

**Assessment**: fp32 and int32 destinations are bit-exact. bfloat8_b is block-quantization noise: ratio spread
centered on 1.000 (no scale bug), max deviation from the host rendering is exactly one quantization step
(some values round differently from the host's quantizer, at the edge of the `bfp8_group` gate). Worth
watching: a second rounding anywhere upstream of the packer would push those values past one step.
**Recommended tolerances**: defaults hold — `helpers.TOLERANCES` (`exact` for every non-bf8b destination,
`bfp8_group` 1 step for bfloat8_b).

## Verifier CLI Summary
Run: `eval/eval_test_runner.sh eval/golden_tests/g6_typecast/` → 44/71 passed, 0 failed/errors/hangs.
`verifier_report.json` copied next to this report.
- supported_pass: 42
- xfail_expected: 27 (every one is a TARGET − SUPPORTED gap covered by Refinement 1 or 2 — see op_requirements.md)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 2 (the two `test_regression.py` tests — untagged by design, both passed)

### Gap coverage (TARGET − SUPPORTED)
| Axis | Missing | Refinement |
|------|---------|------------|
| dtype | uint32, int32 | R1 (RM pairs u32→i32, i32→u32; TILE u32→i32 Q2230 rides along) |
| dtype | float32, uint16 (and int32→bf16) | R2 |
| output_dtype | uint8, uint32 | R1 |
| output_dtype | bfloat16 | R2 |
| layout | ROW_MAJOR | R1 |
| rank | 1, 2 | R1 (every rank-1/2 golden case is ROW_MAJOR) |
No omissions.

## Recommendations
- **Pair gating**: SUPPORTED is per-axis, so after R1/R2 the per-axis product claims pairs with no golden case
  (e.g. uint16→uint8). EXCLUSIONS must stay empty (prompt rule), so the compute kernel must have an arm for
  every pair the per-axis product admits, or the host must reject a pair with no arm with a `ValueError`
  (not a support refusal) — never silently run a wrong arm.
- **Decode-shape occupancy** (design perf lamp "Grid synchronization"): Q2191/Q2216/Q2227 hold 1–16 tiles;
  one tile per core over up to 16 cores may be launch-bound. Unmeasured — candidate for a later perf phase
  once R3/R4 land.
- **Bank-affine / staggered core assignment** (design perf lamp): same bytes, fewer NoC hops; a lever for R4.
- **Unpack-to-DEST for the no-SFPU arms** (bf16→fp32/bf8b): bit results unchanged, may skip the FPU datacopy;
  a lever for R4.
