# Verification Report: layer_norm

Target: Quasar (`quasar`) on the ttsim simulator; Metal 2.0 `ttnn.ProgramSpec`, one `ttnn.generic_op` dispatch
per call. Specification: `eval/specs/layer_norm/spec.py`. Golden results: `/tmp/ln_golden_r1` (report copied
to `pipeline_logs/verifier/verifier_report.json`).

## Code Review

Fixed in place:

1. **Disjoint-lifetime ring with a hand-waved justification (L1 ledger check 4).** `cb_rstd` (rstd → affine)
   and `cb_row_partial` (2Σx → mean reduce, Σc² → var reduce) never hold data at the same time. Each use is
   popped before the next is pushed. The ledger's only reason for keeping them separate was "the per-stage
   debug path". I folded rstd into `cb_row_partial` as its third FIFO use, which takes the op from 12 rings
   to 11 and saves `bh·t_acc` per NEO. The change is in `kernels/layer_norm_compute.cpp` and
   `layer_norm_program_artifacts.py`.
2. **DRY: a duplicated page size.** `Pages.t_sc` was computed with exactly the same expression as `t_acc`. It is
   removed, and the scaler rings now use `t_acc`. The ledger's symbol table had also stated it wrongly, as
   "bf16 tile, 2048 B", while the code allocates the intermediate format; that is fixed too.
3. **DRY: the row-dealing rule written twice.** The reader and the writer each had their own copy of the
   `row_of(neo, round)` lambda, and all three kernels repeated `rounds = ceil(num_rows / C)`. The three
   kernels must agree on these exactly, or a STRIDED ring hangs. Both now live once, in
   `kernels/layer_norm_rows.hpp`, and all three kernels include it.
4. **Ledger currency.** I updated the ledger: the rstd row is gone, the footprint expression now has `2·bh·t_acc`,
   and the worked numbers were recomputed from the code's own tables (for example (1,1,64,4096) is
   1,863,680 / 1,067,008 B per NEO). I added an open bound for L1-resident tensor pages, and a short as-built
   note at the top of `op_design.md`'s ring table.

All 23 local tests and the golden suite were re-run after these fixes: green, with identical categories.

Reviewed and left as they are (correct):

- **Kernel API usage.** `void kernel_main()`, `api/...` include paths and `TensorAccessor` throughout. Reader
  and writer use implicit sync: one tagged transfer per tile and no per-tile barrier, so neither half of the
  dataflow dribbles. The constant rings use explicit sync, listed in `disable_dfb_implicit_sync_for`, and
  their NoC zero-writes run under user trid 0. Both data-movement kernels hand the sticky trid back as their
  last statement.
- **Ring push/pop balance per NEO per block.**
  - `input_block`: Rb·Wt pushed; waited Cumulative in `row_sum`, popped per tile in `centre`.
  - `row_partial`: Rb pushed and popped three times.
  - `row_stat`: Rb pushed and popped twice.
  - `centred`: Rb·Wt pushed; waited without a pop in `sq_sum`, popped in `affine`.
  - Kernel-lifetime rings (γ/β rows, the scalers, ε) are popped once after the last block.
- **Helpers.**
  - Every compute phase is a kernel-library helper: `eltwise_chain` with `BinaryFpu`, `DestReuseBinary`,
    `UnaryBcast` and `Rsqrt`, plus `reduce<SUM, REDUCE_ROW>`.
  - The reduce scalers are written with `prepare_reduce_scaler`.
  - There are two raw calls, both justified in the kernel header:
    `pack_init(<ring>)` and `llk_math_hw_configure`.
  - DEST accumulation plus a one-tile reduce is used, instead of a whole-row reduce
    (the FPU reduce goes through tf32).
- **Broadcasts.** The mean and rstd are applied as Col broadcasts through srcB. γ and β are read from row 0
  only (R8), through the Row broadcast of the expansion. ε is a scalar broadcast of element [0,0] and fills
  no tile. None is redundant.

Deferred to refinements (architectural): none from the code review. Two ledger/budget findings are folded into
refinements (see the L1 Ledger Audit).

## Specification Rules

| Rule | Applies at Phase 0 | Status |
|------|-------------------|--------|
| R1 (float32 + 16-bit DEST refused) | dormant (float32 not supported) | `validate()` raises `UnsupportedAxisValue` before the SUPPORTED check. Re-arms in Refinement 2. |
| R2 (one `default_compute_kernel_config()`) | yes | exported; `None` resolves through it only (`_config_or_default`); lint passes |
| R3 (no gate on fidelity / approx) | yes | `validate()` reads only `fp32_dest_acc_en`; the kernel takes `fpu_math_fidelity` from the caller |
| R4 (device compute config accepted) | yes | the fields are read as attributes and turned into a `ComputeHardwareConfig`; `test_layer_norm_interface.py` passes a `WormholeComputeKernelConfig` for both model calls |
| R5 (subtract the mean, biased variance) | yes | two-pass centred variance, scaler `1/W`. The `uniform` regression case passes (PCC 0.999998). |
| R6 (padding out of sums and count) | dormant (`w_non_aligned` refused) | **the kernel would violate it once the gate opens**. A verifier probe on uniform [0,1) inputs gives PCC 0.955 / rel-RMS 0.44 at W=17, because the padded columns of `centre` hold `−mean` and enter `sq_sum`. Correctly refused today. The mask is Refinement 1's core work. |
| R7 (ε as passed) | yes | ε is written as fp32 bits from a runtime argument (bf16 round-to-nearest-even under 16-bit DEST). The `small_magnitude` case passes. |
| R8 (row 0 of weight/bias) | yes | every cell's weight has 32 distinct rows, and all pass |
| R9 (affine axes off `weight`, bias must match) | yes | `affine_dtype`/`affine_layout` are read off `weight`; a mismatched `bias` raises `ValueError`; the output takes the input's dtype |
| R10 (DRAM or L1 input) | partially (DRAM) | L1 is queued in Refinement 1; probed correct with the gate removed |
| R11 (γ/β read where they are) | yes | `TensorAccessor` on their own buffers; no gate on their placement |
| R12 (output in the input's buffer) | yes | `allocate_tensor_on_device(input.spec)`; the interface test asserts DRAM, interleaved; the L1 probe returned an L1 buffer |
| R13 (`memory_axes`) | yes | `eval.memory.memory_axes(input_tensor)` in `validate()` |
| R14 (`NotImplementedError` outside SUPPORTED) | yes | `UnsupportedAxisValue` / `ExcludedCell` subclass `NotImplementedError`; 522 cells refused this way, 0 wrong-mode |
| X1, X2 (malformed weight/bias) | yes | `ValueError`; both rejection tests pass |

GUIDANCE: followed. That covers the compute-config construction, the two-pass variance, ring sizes derived from
the device rather than from the widest shape, and DEST accumulation instead of a folded reduce. No advisories.

`python3 -m eval.validate_contract layer_norm --phase0 --text`: **PASSED**, with 0 errors and 6 unresolved
items. All 6 are the specification's open items, not defects.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from `eval.specs.layer_norm.spec`; no axis function is defined in the op file.
- `SUPPORTED` equals `PHASE0` on all 9 axes (lint-checked).
- `EXCLUSIONS = []`. The first-phase rectangle has no refused cell.
- `validate()` checks the AXES taggers, then the R1 refusal, then SUPPORTED per axis, then EXCLUSIONS, then the
  malformed-call checks (`ValueError`), then the `row_resident` fit predicate (`NotImplementedError`). The public
  entry point calls it first.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed: no xpass_drift.

**INVALID audit (specification).** `INVALID = []`, which is right for this rectangle:
- No pair of axes is structurally impossible.
- float32 + `fp32_dest_acc_en=False` is legal but lossy, and is handled by rule R1, not by INVALID.
- The canonical bf8b + ROW_MAJOR entry does not apply: `bfloat8_b` is in no dtype axis (Quasar has no path for
  it), and `layout` is TILE-only.
- There is no no-weight canonicalisation: `weight` and `bias` are required by the signature (DECISIONS).

## Design Conformance

- **Algorithm.** It matches the design: a two-pass centred variance on a resident row; DEST-accumulated row sums
  feeding a one-tile reduce; rsqrt(var + ε); and the affine step. The implementer departed from the design's
  realisation (not its math) because of two Quasar limits, both documented in the compute kernel's header:
  - ε is added by the FPU as a scalar broadcast, because a 32-bit `UnaryBcast` is refused;
  - rstd is broadcast through srcB, and γ is expanded like β.
- **Pipeline topology.** Reader → compute → writer on separate harts, as designed. The stall-shadows are in
  place: the Cumulative wait makes `row_sum` run inside the read; β (and now γ) is read before x; and
  `centre` pops the input block, so the reader refills it under passes 3–6.
- **Work distribution / thread plan.** Read off the host program: one named constant, `THREAD_PLAN = (4, 4, 2)`,
  is clamped per call by `select_blocking()`. The effective counts reach the kernels as compile-time arguments
  and set `KernelSpec(num_threads=…)`.
  - Rows are dealt whole rounds to clusters (`split_work_to_cores(row_wise=True)`) and round-robin to NEOs.
  - All rings are STRIDED, with capacity counted per NEO.
  - The design states the op is **compute-bound**, so C = 4 is right.
  - `test_layer_norm_plan.py` covers the plan:
    - (a) every thread ran, at (4, 4, 2) on one cluster, including a padded round;
    - (b) C = 1, 2 and 4 are each selected and value-checked;
    - (c) the op after `ttnn.experimental.quasar.add`, and the op before `ttnn.to_layout(..., ROW_MAJOR)`, in one
      device session, at C = 1 and C = 4.

    All pass. This backend has no timing model, so the plan was not measured against the alternatives named in
    the design's perf lamps ((2,4,2), (1,4,1), NEOs-first packing). That stays open for a timed backend.
- **Grid fill.** For shapes with more rows than clusters it fills all 32 clusters. For the model's own few-row
  shapes it engages 2 or 4 NEOs out of 128. This follows the design's chosen regime (`row_resident`, with
  `w_split_combine` deferred, reason recorded) and is queued as perf Refinement 3, not treated as a
  design-contract violation.
- **Blocking-model fidelity.** `block_tile_rows`, `INPUT_BLOCK_DEPTH`, `OUTPUT_DEPTH`, `AFFINE_STREAM_DEPTH`
  and the plan are each defined once. Every ring's capacity comes from one table (`ring_entries_per_neo`), and
  the same table drives both `fits()` and the `DataflowBufferSpec`s.
  - No collapsed knob. `cb_input_block` and `cb_centred` scale with `bh·Wt`; that is a predicate-guarded
    resident set with a stated bound and a `NotImplementedError` fallback (row streaming is deferred because no
    TARGET shape reaches it).
  - The per-core loop iterates blocks of `block_tile_rows` with a ragged last block, so the knob is not
    half-turned.
  - Expression: the reader and writer issue per tile but with no per-unit completion boundary (implicit sync).
    Compute chains run on `grid(Rb, Wt)`. No expression failure.
- **Inter-core communication.** None, by design. The combine and fan-out regimes are deferred and queued.

## L1 Ledger Audit

- **Currency.** Every one of the 11 declared rings has a row, and every row is a live ring. The size
  expressions match `ring_entries_per_neo × ring_page_bytes`, and the worked numbers were recomputed from the
  code. The ledger was fixed for: the folded rstd, the `t_sc` misstatement, and the footprint expression.
- **Capacity vs live set.**
  - *Over*: none. `cb_affine_stream` and `cb_output_tiles` are depth-2 windows with a live set of 1; that is
    stated double-buffering. `cb_epsilon` is a whole tile with one meaningful element, because a scalar
    broadcast reads a tile page.
  - *Under*: none. `cb_input_block` and `cb_centred` span both `tile_row` (bh) and `w_tile` (Wt), and their
    capacity scales with both. The per-row rings span `bh` only.
- **Page format vs DEST width.** All intermediates follow `t_acc`: Float32 when `fp32_dest_acc_en`, bf16
  otherwise. No over or under case. γ/β rows are widened to `t_acc` under fp32 DEST, for a mechanism reason
  (the affine chain has one ALU format on Quasar).
- **Disjoint lifetimes.**
  - `cb_rstd` / `cb_row_partial`: **fixed** (folded).
  - `cb_affine_stream` / `cb_output_tiles`: now justified in the ledger (they have different producer and
    consumer kernels).
  - The rest are concurrent, with reasons stated.
- **Bounds.** Wt, bh, C and the page sizes have bounds and predicates.
  - **Open bound, folded into Refinement 1:** `l1_budget` does not subtract L1-resident tensor pages. That is
    harmless while `memory_config = L1` is refused, but it must be fixed or proven before L1 is claimed (the
    largest INPUTS cell holds 128 KiB per bank in float32 in + out).
- **Block-size defaults.** The interleaved default holds: rows are spread over the full grid first, then the
  coarsest block (`bh = rounds`) is taken whenever it fits. No departure from the default.
- **Per-NEO footprint.**

  ```
  F(bh) = bh·Wt·(t_in + t_acc) + 2·Wt·t_acc + 2·t_aff + 2·t_in + 2·bh·t_acc + 3·t_acc
  ```

  The `bh·Wt` term scales with the block and the row, the `Wt` term with the row, and the `bh` term with the
  block; the rest is constant. Per cluster the constraint is `C·F(bh) ≤ l1_budget`, and every ring is capped
  at ≤ 1,048,560 B.
- **Data-movement budget.** It is present and consistent with the code:
  - x crosses DRAM once and the output once (plus at most C−1 duplicate rows from the padded round);
  - γ and β cross E× (once per engaged NEO).

  The cheapest-traffic split, `w_split_combine`, is a deferred regime with a delta. Its stated reason is that
  `row_resident` is needed as its per-slice kernel, and that the combine has no worked mechanism on Quasar.
  That is a build-order reason, not "unreachable". It is now **queued** as Refinement 3, and the γ/β E× term
  as Refinement 4.

## Precision Baseline

`tests/ttnn/unit_tests/operations/layer_norm/test_layer_norm_precision_baseline.py`: bf16 input/weight/bias,
TILE, DRAM, ε = 1e-6, seed 1234, HiFi4. PCC is from `comp_pcc`, max abs error from `comp_allclose`, and relative
RMS is the RMS error over the reference's std. The ratio is got/true over the elements with |ref| > 0.1% of the
maximum.

| Shape | DEST | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 (bf16) | ratio p5 / median / p95 |
|-------|------|-----|-------------|--------------|------------------|----------------|-------------------------|
| (1,1,32,64) | fp32 | 0.999998 | 0.0232 | 0.00158 | 0.00186 | 4 | 0.9959 / 0.9997 / 1.0029 |
| (1,1,32,64) | bf16 | 0.999993 | 0.0857 | 0.00287 | 0.00371 | 16 | 0.9896 / 0.9999 / 1.0093 |
| (2,1,512,512) | fp32 | 0.999998 | 0.0398 | 0.00165 | 0.00177 | 3 | 0.9966 / 0.9998 / 1.0029 |
| (2,1,512,512) | bf16 | 0.999994 | 0.1068 | 0.00303 | 0.00361 | 15 | 0.9902 / 0.9996 / 1.0074 |
| (1,1,128,1024) (model) | fp32 | 0.999998 | 0.0341 | 0.00160 | 0.00176 | 3 | 0.9966 / 0.9998 / 1.0029 |
| (1,1,128,1024) (model) | bf16 | 0.999994 | 0.0903 | 0.00296 | 0.00362 | 16 | 0.9901 / 0.9997 / 1.0077 |
| (1,1,64,4096) (model) | fp32 | 0.999998 | 0.0280 | 0.00160 | 0.00176 | 3 | 0.9966 / 0.9998 / 1.0029 |
| (1,1,64,4096) (model) | bf16 | 0.999993 | 0.0941 | 0.00351 | 0.00431 | 20 | 0.9918 / 1.0009 / 1.0133 |

**Assessment.** Errors are at bf16 output rounding with fp32 DEST, and a few bf16 steps beyond it with the
16-bit accumulator. The 16-bit accumulator's error grows only mildly with W: ULP p99 is 16 at W=1024 and 20 at
W=4096, and rel-RMS 0.0036 → 0.0043. That is two orders of magnitude inside the 0.999 / 0.13 gate. The ratio
medians sit on 1.000 ± 0.001, so there is no scale or structural bug on the supported cells. The golden cells
agree: PCC ≥ 0.999992 everywhere, and the numeric regressions (`small_magnitude`, `uniform`) give PCC ≥ 0.999998.

**Recommended tolerances.** The defaults hold: (bf16, 16-bit DEST) = the specification's legacy (0.999, 0.13),
and (bf16, fp32 DEST) is graded by `NUMERIC_RESOLVED` `('bfloat16', 'default')`. Both pass with wide margin.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/layer_norm/` (551 tests, 8 sim workers, ~48 s), then
`python3 -m eval.verify_supported … --queue op_requirements.md`:

- supported_pass: 26 (22 PHASE0 matrix cells + 4 numeric regression cases)
- xfail_expected: 522
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3. These are rejections X1 and X2 (both passed) and the empty loose-case parameter (skipped:
  `LOOSE_CASES = []`).
- queue_gaps: 0. Every value of TARGET − SUPPORTED is named in `op_requirements.md`:
  - alignment {w_non_aligned, h_non_aligned}, rank {2, 3} and memory_config L1 in Refinement 1;
  - dtype float32 and affine_dtype float32 in Refinement 2.

Local tests: `tests/ttnn/unit_tests/operations/layer_norm/` has 31 tests, all passing: acceptance 7, interface
2, plan / threads / session 14, and precision baseline 8.

## Recommendations

- **Queue order.**
  1. Refinement 1: alignment mask + ranks + L1.
  2. Refinement 2: float32.
  3. Refinements 3–5: perf, structural gates.

  No loose case is perf-flagged. The perf work is free-selected and starts with the model's own two calls,
  which today use 2–4 of 128 NEOs.
- **R6 trap for Refinement 1.** The golden w_non_aligned cells use standard-normal data, which hides a missing
  mask (the row means are near 0). Require the uniform-input test named in Refinement 1's Done-when.
- **The float32 γ/β expansion.** The current `UnaryBcast<Row>` prologue cannot take a 32-bit page on Quasar.
  Refinement 2 has to replace it; reader-side replication of row 0 is the option that also serves the perf
  lamp.
- **Timed-backend follow-ups (not refinements; this board has no clock).** On `ttsim-perf` or hardware, measure:
  - the thread-plan alternatives ((2,4,2), (1,4,1));
  - packing NEOs first against spreading over clusters first;
  - the structural wins of Refinements 3–5.
- **Memory pressure (no OOM today).** Under fp32 DEST, `cb_centred` reaches the 1 MiB ring cap at Wt = 256
  (W = 8192), so `fits(1,1)` fails there and `validate()` refuses the call by name. INPUTS stops at W = 4096,
  where the largest ring is 512 KiB.
