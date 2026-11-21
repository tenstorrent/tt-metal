# Operation Requirements: typecast

## Definition
- **Formula**: `output[i] = convert_{input.dtype -> dtype}(input[i])` — float→float: rounded once (RNE); float→integer: truncate toward zero, then saturate (R1); integer→float: exact value rounded once; integer→integer: wrap modulo 2^bits (R2); one step, no third dtype on the way (R3). Same dtype: a copy into a fresh tensor.
- **PyTorch Reference**: `input.to(dtype)` (the specification's `reference(input_tensor, dtype, axes)` in float64: the value itself for a float output; `trunc().clamp(lo, hi)` for float→int; `remainder` wrap for int→int)
- **Import Path**: `from ttnn.operations.typecast import typecast`
- **Function Signature**: `typecast(input_tensor: ttnn.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor` (`dtype` positional or `dtype=`; not a `ttnn.DataType` → `TypeError`, REJECTIONS X1)
- **Specification**: `eval/specs/typecast/spec.py` (the only source of requirements). Target: Quasar on ttsim (`.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **One dispatch (run rule)**: every path stays exactly one `ttnn.generic_op` of one `ProgramSpec`. A manipulation-op wrapper (`ttnn.to_layout`, `ttnn.typecast`, `ttnn.tilize`, …) before or after the program is a second dispatch and is **not** available even as a partial-tick escape hatch on this op.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED target_dtype**: [bfloat16, float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: DRAM / interleaved
- **SUPPORTED shape-derived axes**: alignment = tile_aligned; rank = 4
- **Regime built**: `tile_stream_float` — reader (implicit sync, tagged reads) → `cb_input_tiles` → compute (`compute_kernel_lib::typecast`, 32-bit DEST; UnpackToDest for fp32, UnpackToSrc for bf16) → `cb_output_tiles` → writer (implicit sync)
- **Cores**: full 8×4 grid once `tensor_tiles ≥ 32` (`split_work_to_cores`, row-wise, contiguous tile ranges)
- **Thread plan**: (R, C, W) = (1, 1, 1), `THREAD_PLAN` — the float pairs are bound by data movement (no SFPU work for bf16↔fp32 on Quasar); other plans tested bit-exact
- **Blocking**: `block_tiles = clamp(ceil(core_tiles_max / BUFFER_DEPTH), 1, cap)`, cap = one DEST section (4 tiles); `BUFFER_DEPTH = 2`
- **Compute config**: none exposed (spec: the op has no compute config); fixed LoFi, 32-bit DEST, half-sync
- **Golden baseline**: 24 / 24 supported cells pass; 2285 xfail_expected; 832 skipped (224 INVALID + bfloat8_b unbuildable on quasar); 0 supported_fail / xpass_drift / xfail_wrong_mode
- **Precision**: bit-exact on all four pairs (fp32→bf16 = torch RNE, ≤ 0.5 ULP)

### [ ] Refinement 1 — Integer dtypes on the tile path, plus the tile path's placement and shape surface

**Goal**: add `ttnn.int32`, `ttnn.uint32`, `ttnn.uint16`, `ttnn.uint8` to SUPPORTED["dtype"] (int32, uint32, uint16, uint8) and to SUPPORTED["target_dtype"] (int32, uint32, uint16, uint8) — every pair among {float32, bfloat16, int32, uint32, uint16, uint8}, built as the design's `tile_stream_int` regime on top of `tile_stream_float` (same reader / writer / split; the compute call's formats, the ring formats and the per-ring unpack mode change). In the same refinement, widen the tile path's free knob-turns: add `"L1"` to SUPPORTED["memory_config"] (L1, interleaved — R4; the output follows the input's buffer, R6), add 2 and 3 to SUPPORTED["rank"] (rank 2, 3), and add `w_non_aligned`, `h_non_aligned` to SUPPORTED["alignment"] (padded tiles are pages like any other on the tile path).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Anchor for Refinement 3: this refinement lands the full contract of the perf target (an SFPU integer pair, TILE, DRAM, rank 4, tile-aligned), so it must ship performant, not correct-only: keep Phase 0's full-grid contiguous split, the double-buffered rings and the per-block compute sync on the integer path. The thread plan stays (1, 1, 1) here; turning it is Refinement 3.
- The skill is written for float dtypes; the integer-specific questions below are outside it and are this refinement's to settle, in this order (the design's `tile_stream_int` regime row and Risk 11 list them):
  1. **Ring formats**: uint16 and uint32 are not DFB formats. A ring carried under another format of the same width changes what the unpacker decodes (a uint16 ring declared bfloat16 and unpacked through SrcA is decoded as bf16), so the carried format and the unpack mode must preserve the bits for every source and sink.
  2. **Unpack modes**: Int8 / Int16 / UInt8 sources must UnpackToDest on Quasar (`typecast.h` static_assert); 32-bit sources UnpackToDest (SrcA keeps 19 bits); a bfloat16 page must not UnpackToDest. `DataFormatReconfig` stays Disabled on any ring read with UnpackToDest.
  3. **Semantics vs. the SFPU arms**: the Quasar `typecast_tile` list has no Int32↔UInt32, Int32→UInt8 or UInt32→UInt8 arm, and an existing int→int arm may saturate where R2 requires a wrap (int32 300 → uint8 44, −1 → 255). Same-width int↔int (int32↔uint32) is a bit reinterpretation; narrowing int→int is a mask of the low bits. R3 (one step: an int32 never passes through bf16) is checked by the int32→uint32 cells. R1 (float→int saturate, truncate toward zero) is checked by the four NUMERIC regression cases (`beyond_uint8`, `beyond_uint16`, `beyond_int32`, `fractions`), which become live in this refinement and must pass.
  4. Every integer pair is graded `exact` (int→bf16 rounds RNE once); only the int→bfloat8_b pairs carry PCC 0.99, and those are unbuildable here.
- EXCLUSIONS only for a structural gap that produces no interpretable metric (severity=bug); a cell failing `numerical-precision` stays failing and gets a follow-up refinement. No new axis.
- The placement and shape widening needs no kernel change on the tile path (the TensorAccessor addresses DRAM and L1 alike, and `buffer_num_pages()` already counts padded tiles per image). It still needs evidence: the golden cells, and a test that an L1 input returns `output.memory_config()` == L1 interleaved (R6, a review rule the golden suite does not compare). Once L1 is in SUPPORTED, a sharded L1 input must still be refused by the `sharding` axis by name (R7); `test_sharded_input_refused_by_name` already pins that.

**Done when**: every golden cell in the tile layout with both dtypes in {float32, bfloat16, int32, uint32, uint16, uint8}, in DRAM or L1, of rank 2–4 and any alignment, passes (bfloat8_b cells stay skipped as unbuildable on quasar); the four NUMERIC regression cases pass; `verify_supported` shows 0 supported_fail / xpass_drift / xfail_wrong_mode; all prior tests pass.

### [ ] Refinement 2 — Row-major layout

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to SUPPORTED["layout"] (ROW_MAJOR in, ROW_MAJOR out) for every dtype pair SUPPORTED holds at that point (float and integer), every rank (2–4), alignment (tile_aligned, w_non_aligned, h_non_aligned) and placement (DRAM, L1) — the design's `row_major_stream` regime, natively in the kernels: the reader fills each ring entry from contiguous stick bytes, compute converts the entries as it does tiles (the conversion is position-agnostic), and the writer emits sticks.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**:
- Ordered after Refinement 1 because the anchor fixes Refinement 1 to the perf target's contract. This is the harder refinement (a reader/writer rewrite and a new page unit), and SUPPORTED's axes are independent, so it owes the RM path for every dtype Refinement 1 added, integer ones included.
- `/interleaved-parallel` rides along: the RM path must keep the full-grid, contiguous split (ranges of sticks / entries per cluster, `row_wise=True`). A single-core RM path is a design-contract violation, not a later refinement.
- What makes RM hard here: input and output entries hold the same element count at different byte widths (bf16 2 B, fp32/int32/uint32 4 B, uint16 2 B, uint8 1 B). Non-aligned sticks are not multiples of the L1 (16 B) or DRAM (64 B) alignment: W=17 bf16 is 34 B, and W=17 uint8 is 17 B. Rank 2 and 3 sticks come from flattening every dimension but the last. A ragged last entry converts padding that must not be written.
- An entry assembled from several stick reads is explicit sync: list the ring in `disable_dfb_implicit_sync_for`, tag every transfer (`{.trid = 0}`, barrier on the same id), and hand the tag back as the last statement (target page *Ring protocols*). Re-run the device-session test (another program, then this op, then `ttnn.to_layout`) on an RM cell.
- The spec's RM input → bfloat8_b output (a tilize) is unbuildable on quasar. Nothing is owed for it here.
- No manipulation-op wrapper as a fallback (one-dispatch rule above). If the RM path cannot be finished in one pass, partial-tick: put the RM cells it does not yet cover in EXCLUSIONS (cell-level, e.g. `{"layout": ttnn.ROW_MAJOR_LAYOUT, "dtype": ttnn.uint8}`), only for a structural gap, and file `Refinement 2b` right after this one.

**Done when**: every golden cell with `layout=ROW_MAJOR` passes for all SUPPORTED dtype pairs, ranks, alignments and placements (bfloat8_b cells skipped as unbuildable, INVALID cells skipped); output layout is ROW_MAJOR; the device-session test passes on an RM cell; `verify_supported` loud categories are 0; all prior tests pass.

### [ ] Refinement 3 — Thread plan for the SFPU-bound integer conversions (and the RM path's data-movement harts)

**Type**: perf

**Goal**: no LOOSE_CASES entry is perf-flagged, so the region is chosen here: the integer pairs that run the SFPU `typecast_tile` per tile (float↔int and int↔int, unlike bf16↔fp32, which is an SFPU no-op). Target shape: `[1, 1, 2048, 256]` (512 tiles, 16 per cluster on the 8×4 grid), `float32 → int32` and `int32 → float32`, TILE, DRAM, interleaved, rank 4, tile-aligned (no compute config: the op exposes none). Those cells keep Phase 0's (1, 1, 1) plan. Levers: the thread plan per regime (C = 4 for the SFPU integer pairs; (1, 1, 1) stays for the float pairs), plus R = W = 2 on Refinement 2's RM path if its entries are explicit-sync assemblies. See `ttnn/ttnn/operations/examples/master.md` and `.claude/references/quasar_op_examples/README.md`. The plan stays one named source on the host (a per-regime table is fine), and its counts reach the kernels as CT args. No SUPPORTED change.

**Verifier notes**: depends on Refinement 1 (the integer path) and, for the RM half, on Refinement 2. This backend has no clock (`.claude/references/backends/ttsim.md`, *Performance measurement*), so the gate is a structural count, and the gain in time stays unconfirmed until `ttsim-perf` or hardware measures it. Never report a simulator timing as a speedup. The ring capacity becomes `BUFFER_DEPTH · block_tiles · C`, and `choose_block_tiles` already divides the per-core share by C.

**Done when**: (a) structural count, read off the `ProgramSpec` by a test: compute threads at work on the target cells go from 1 to 4, with float-pair cells still at (1, 1, 1); on an explicit-sync RM path, reader + writer harts go from 1 + 1 to 2 + 2. (b) A test tags output by NEO (or otherwise shows that each of the 4 NEOs converted its own tiles), as the target page requires: a correct result alone does not show the threads ran. (c) The golden suite stays green (0 loud categories). (d) No regression across the config-spanning guard set: one representative per distinct kernel path × layout × placement (float TILE DRAM, float TILE L1, integer TILE, RM float, RM integer) stays bit-exact. (e) Before/after counts are recorded in the changelog, stating that the gain in time is unconfirmed on ttsim.

## Omitted

- Omitted: SUPPORTED["dtype"] bfloat8_b — no block-float path on quasar (target page *Family facts*); its cells skip as `unbuildable on quasar`, and the target page says to queue no refinement for them.
- Omitted: SUPPORTED["target_dtype"] bfloat8_b — same reason (no block-float path on quasar; unbuildable cells).
