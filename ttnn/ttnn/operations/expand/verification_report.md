# Verification Report: expand

Target: Quasar (`quasar`) on the ttsim simulator. Specification: `eval/specs/expand/spec.py` (lint: PASSED, 0
errors, 5 UNRESOLVED open items, all in sections the op does not act on: `tt_regen_spec.governs`,
`NUMERIC.special_values`, `NUMERIC.freedoms`, `PERFORMANCE.bar`, `PERFORMANCE.measure`).

## Code Review

The implementation matches `op_design.md` (regime R-H) closely. The review found no correctness defect. Fixed:

1. **Ring capacity above its live set (L1 ledger audit 2).** The ring was always `buffer_depth · block_src_tiles`
   entries (depth 2). By default every cluster runs exactly one block (the block is all of the busiest cluster's
   sources), so the second half was never live: 2× the live set on every Phase 0 INPUTS entry. Now
   `depth = min(buffer_depth, max_blocks_per_core)`, computed on host from the same knobs. `buffer_depth` is still
   the one source of the depth knob: the clamp is derived, like the existing `block_src_tiles_cap`, so a refinement
   that makes ≥ 2 blocks per cluster (Refinement 5) gets the double buffer back with no further change.
   Traced call: 12,288 B → 6,144 B. The knob test that turned `buffer_depth=3` now uses `block_src_tiles=1`
   (3 blocks per cluster), so a 3-deep ring is still run. `l1_ledger.md` updated.
2. **R7: refused by name.** `validate()` raised on the *first* axis outside SUPPORTED, so a sharded L1 input was
   refused for `memory_config=L1` and the `sharding` axis never appeared in the message. `validate()` now names
   every axis outside SUPPORTED in one `UnsupportedAxisValue` (order SUPPORTED → EXCLUSIONS unchanged).
3. **Review-check rules had no test.** Added to `test_expand_plan.py`: `test_output_placed_like_input` (R6: output
   `memory_config()` equals the input's, DRAM, interleaved), `test_sharded_input_refused_by_name` (R7: a
   height-sharded input raises `NotImplementedError` naming `sharding`), `test_ring_depth_structural_count` (the
   ring holds `min(buffer_depth, blocks) · block` entries for the default and three forced blocks).

Checked and found conformant (nothing to change):

- **Specification rules.** R1 (byte copy, no format conversion: bit-exact, special values included), R2 (`-1` and
  equal sizes read as kept before the support check), R3 (list and tuple; the loose case passes), R4 (rows 1..31 of
  each input tile overwritten from row 0 before the push; only row 0 reaches the output), R5 (L1 refused for now,
  queued in Refinement 4), R6 (output spec takes `input_tensor.memory_config()`), R7 (`eval.memory.memory_axes`),
  R8 (`UnsupportedAxisValue` / `ExcludedCell` are `NotImplementedError`). X1–X3 raise `ValueError`; a size of 0
  raises `NotImplementedError` (unspecified, per DECISIONS). No rule has a `structure` part.
- **One dispatch.** Host validation, one `allocate_tensor_on_device`, exactly one `ttnn.generic_op`.
- **Algorithm / topology.** Read once, build once, write many; reader → one DM→DM ring → writer; no compute kernel.
- **Work distribution.** `split_work_to_cores(grid, total_out_tiles, row_wise=True)` over the whole queried grid
  (32 clusters on the traced call, 105 output tiles each); source-major contiguous runs; no idle-silicon defect.
- **Thread plan.** Read off the host program: `THREAD_PLAN = (1, 0, 1)`, one named constant; `num_threads` of reader
  and writer come from it and reach the kernels as CT args (`reader_threads`, `writer_threads`, static-asserted to
  1). The design states the op as data-movement bound with the reason (zero compute per tile; output DRAM writes
  dominate), which is the case where the target page's *Choosing the plan* gives one thread of each kind; no NEO is
  used because nothing is computed. The plan's tests exist and pass: per-cluster structural count of input reads,
  `test_after_another_program`, `test_before_a_tt_metal_program` (expand then `ttnn.to_layout(..., ROW_MAJOR)` in one
  session). ttsim has no timing model, so the plan was not measured against W = 2; the
  writer here is explicit sync by necessity (it writes one entry many times) — that is Refinement 3.
- **Ring protocol.** Explicit sync on both ends, ring listed in both kernels' `disable_dfb_implicit_sync_for`,
  every NoC transfer tagged `{.trid = 0}` with barriers on the same id, both kernels reset the sticky tag last.
  Push and pop counts balance (nominal `block_src_tiles` per block on both sides, same block loop).
- **Blocking-model fidelity.** `block_src_tiles`, `buffer_depth`, `max_cores` are host parameters defined once;
  ring capacity and both kernels' loop bounds derive from them (CT `block_src_tiles`); no CB scales with a whole-op
  dimension. Expression: reader per block = 1 reserve, `n` reads, 1 barrier, 5 replication steps each batched over
  the block, 1 push; writer per block = 1 wait, one write per output page, 1 barrier, 1 pop. No per-unit handshake
  in either kernel. The per-core loop works on the whole block, not one tile.
- **API usage.** `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor`, `dataflow_kernel_lib::local_addr`
  for the self-aimed source. The helper alternatives the design rejected (compute copy, stateful local copies,
  mcast, column/fill helpers) are rightly rejected: none expresses a DRAM fan-out or a multi-size row doubling.
- **Broadcast efficiency.** The broadcast tile is built once per source per cluster and written many times; nothing
  is re-filled.

Deferred (architectural, queued): none beyond the refinements below.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification; no op-local axis function.
- `SUPPORTED` equals `PHASE0` axis for axis (dtype, layout, memory_config, sharding, changes, alignment, rank).
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED, then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`; it is the first
  call of `expand()`. Shape-argument errors (X1–X3) are raised before the axes are derived, as the AXES contract
  requires (`(input.shape, output_shape)` after the check).
- The op file declares no `INVALID`.
- No XPASS drift; no auto-fix to SUPPORTED was needed.
- **INVALID audit (specification).** `INVALID = []`. Correct: TARGET has no block-float dtype, so the canonical
  `bfloat8_b + ROW_MAJOR` entry has nothing to apply to; no cross-tensor entries (one tensor); not a norm-like op.
  Every TARGET cell is buildable on Quasar.

## L1 Ledger Audit

- **Currency.** One ring, `cb_row_bcast_tiles`, matches the code: entry `page_size`, `depth · block_src_tiles`
  entries (updated for the depth clamp), input dtype page.
- **Capacity vs live set.** Over: the depth-2 ring with one block per cluster — fixed in place (see Code Review 1).
  Under: the live set spans the source axis (`block_src_tiles`) and capacity scales with it; it streams over the
  expanded axis with no pages for it. No collapsed extent.
- **Page format vs DEST.** Dataflow-only payload, never unpacked; no DEST involved. Correct.
- **Disjoint lifetimes.** One buffer; nothing to share.
- **Bounds.** `block_src_tiles ≤ block_src_tiles_cap = floor(min(1,048,560 B, l1_budget) / (buffer_depth ·
  page_size))`; `depth ≤ buffer_depth`; no op dimension in a capacity expression. Closed form:
  `footprint = min(buffer_depth, max_blocks_per_core) · block_src_tiles · page_size`, scaling with the source block
  knob, the depth knob and the dtype's page size; independent of H_out and of the output size.
- **Data-movement budget.** Present and consistent with the split the code implements (input page reads ≤
  `num_sources + engaged_clusters − 1`, checked by `test_input_reads_structural_count`; output written once; no
  cross-core traffic). The cheapest-traffic split (R-H-colalign) is a deferred regime row with a positive reason
  (+20 % critical path on the traced call for a 0.5 % byte saving, and a core cap at `num_sources`).
- **Block-size defaults.** Held: interleaved, whole grid engaged, coarsest block that fits (all the busiest
  cluster's sources, one block per cluster). Worked footprint on quasar bf16: traced call 6,144 B; largest PHASE0
  entry `(1,8192)→(17,-1)` 16,384 B.

## Precision Baseline

`tests/ttnn/unit_tests/operations/expand/test_expand_precision_baseline.py` (standard-normal bf16 inputs, the
specification's `reference` in float64; `comp_pcc` / `comp_allclose` from `models.common.utility_functions`):

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP (bf16) | got/true ratio p5 / p50 / p95 |
|-------|-----|-------------|--------------|------------------|------------|-------------------------------|
| (1,64) → (32,-1) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,50) → (100,-1) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,2560) → (1330,-1) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,8192) → (17,-1) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |

Every output is bit-identical to `torch.Tensor.expand`. `test_special_values_bit_exact`: NaN (payload kept), ±inf,
±0.0, bf16 subnormals and the largest finite values are copied bit for bit (the specification leaves special values
UNRESOLVED; the op propagates them unchanged by construction).

**Assessment**: exact. A byte copy with no format conversion; no scale or structural signature (ratio exactly 1).
**Recommended tolerances**: exact (`NUMERIC_RESOLVED`: `exact` for bfloat16 and float32).

## Verifier CLI Summary

Golden suite `eval/golden_tests/expand/` (260 tests, 8 sim workers): 12 passed, 248 xfailed, 0 failed, 0 hangs.
`verifier_report.json` (copy in this directory):

- supported_pass: 9 (8 PHASE0 cells + loose case `output_shape_as_list`)
- xfail_expected: 248
- invalid_skipped: 0
- no_axes_found: 3 (the rejection tests X1–X3, all passing)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Queue check (`--queue op_requirements.md`): 8 values of TARGET − SUPPORTED, 0 gaps.

Unit tests `tests/ttnn/unit_tests/operations/expand/`: 38 passed (acceptance 4, plan/knob/one-session/review 29,
precision 5).

## Recommendations

- **Queue.** Refinements 1–2 generalize the tile path's index map and build modes (`leading`, `added`, ranks 3/4,
  then `w`), Refinement 3 is the writer-throughput perf phase on the traced call, Refinement 4 adds row-major,
  float32 and L1 together on the general index map, Refinement 5 is the read/write-overlap perf phase on fan-out-1
  shapes. Perf phases are gated on structural counts; their time gain stays unconfirmed until `ttsim-perf` or
  hardware measures it.
- **Perf lamps not filed.** *Grid synchronization* (fewer clusters for tiny outputs, e.g. `(1,128)→(1024,-1)` reads
  32 input pages for 4 sources) trades input reads against occupancy; no structural count decides it, so it waits
  for a timed backend. *Input read granularity* (read 64 B of the 2 KiB input page) matters only at fan-out 1 and
  needs the DRAM-alignment check; Refinement 5 may take it if its other levers leave the reads on the critical path.
  *R-H-colalign* stays deferred per the ledger.
- **Timed check owed.** The thread-plan choice (1, 0, 1) and every perf refinement here should be re-measured on
  `ttsim-perf` (model cycles) or hardware; this run's backend cannot.
- **Specification open items** (no action for the op): special values are copied unchanged, which a future
  `NUMERIC.special_values` decision can rely on; `PERFORMANCE` has no bar yet.
