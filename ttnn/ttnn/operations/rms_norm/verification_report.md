# Verification Report: rms_norm

Phase-0 verification of the row-parallel, bounded two-pass streaming-reduce
implementation. Reviewed against `op_design.md` (incl. the Blocking Model),
`eval/prompts/rms_norm.txt`, the golden feature spec, and the kernel-lib helpers.

## Code Review

### What was reviewed (no code changes needed in the op/kernels)

The implementation is faithful to the design and clean. I read the full op file,
program descriptor, and all three kernels against the design's binding
dimensions and the blocking model, and found **no fixable defects in the op or
kernels**. Details:

- **Registry declarations wired correctly.** `INPUT_TAGGERS` (`tag_alignment`,
  `tag_rank`, both with the `(inputs, axes)` signature), `SUPPORTED` (all 9 axes
  the kernel gates on), `EXCLUSIONS` (empty — justified, see below), and
  `validate()` (per-axis SUPPORTED then cell-level EXCLUSIONS, raising
  `UnsupportedAxisValue` / `ExcludedCell` from `_op_contract`). The public
  `rms_norm()` calls `validate()` as its first statement. The op file does **not**
  declare `INVALID` (correct — that lives in `feature_spec.py`).
- **Blocking-model fidelity — clean.** Every block knob is a live parameter with
  a single source of truth: `BLOCK_SIZE = _pick_block_size(Wt)` computed once on
  the host and passed as one CT arg to reader+compute; `NUM_BLOCKS = Wt //
  BLOCK_SIZE` derived; `DEPTH = 2` a host constant; grid a runtime-arg split via
  `split_work_to_cores(..., row_wise=True)`. **No CB is sized by an op dimension**
  (`Wt`/`W`/`H`/`R`) — every page count is `DEPTH*BLOCK_SIZE`, `2*BLOCK_SIZE`, or
  a constant, and `cb_rstd` is 1 tile/row. No collapsed knob, no duplicated block
  literal, no CB that unconditionally scales with the reduce dim. The `2` in the
  `2*BLOCK_SIZE` intermediates (`cb_xsq`, `cb_norm`) is the design's "hold a full
  block" rule, not a duplicated `DEPTH`.
- **Performance conformance — grid filled, both dataflow halves batched.** Row
  work is spread across the whole compute grid; the TILE reader coalesces
  `BLOCK_SIZE` reads behind one barrier and the TILE writer mirrors it; CBs are
  double-buffered (`DEPTH=2`). This matches the design's "multi-core from day 1"
  and "keep BLOCK_SIZE reads in flight". (Single-core under-fill at `R=1` /
  decode is the known lamp-2 case, queued as a refinement, not a bug.)
- **Helper usage — full, with two justified library-bug deviations (verified).**
  Compute uses `square`, `reduce<>`, `transform_in_place`, `mul<…,Col/Row,…>`,
  `copy`, `tilize`/`untilize`; dataflow uses `read_sticks_for_tilize`,
  `write_sticks_after_untilize`, `prepare_reduce_scaler`, `TensorAccessor`.
  Two streaming-reduce **wrapper** helpers are genuinely stale in this
  kernel-lib checkout and the kernel correctly calls the working underlying
  helpers instead:
    1. `accumulate_reduce_block<>` / `accumulate_reduce<>`
       (`streaming_reduce_helpers.inl:32`) pass the CB ids as **runtime** args to
       `reduce<>`, but the current `reduce<>` (`reduce_helpers_compute.hpp:522`)
       takes them as **template** params — the wrapper does not compile. The
       compute kernel calls `reduce<SUM,REDUCE_ROW,cb_xsq,cb_scaler,cb_rstd>(…,
       Accumulate::at(cb_rstd,b), …, ps)` directly with last-block partial-scaler
       routing — exactly what the wrapper would emit.
    2. `prepare_partial_reduce_scalers<…>` (`reduce_helpers_dataflow.inl:349/351`)
       forwards a 4th `compute_uses_reduce_tile` template arg to
       `prepare_reduce_scaler<>`, which takes 3 — does not compile. The reader
       emits the same full(tile0)+partial(tile1) pair via two direct
       `prepare_reduce_scaler` calls.
  Both deviations still route through the intended helpers, not raw LLK, and both
  are documented in-kernel. **These are kernel-lib bugs, not op bugs** — see
  Recommendations for the upstream note. I did not revert to the broken wrappers.
- **Includes / syntax / API.** `api/dataflow/dataflow_api.h` and
  `api/compute/*` (not bare includes); `void kernel_main()` (not the deprecated
  namespace form); `TensorAccessor` (not `InterleavedAddrGen`). All correct.
- **Broadcast dims.** `x·rstd` uses `BroadcastDim::Col` (REDUCE_ROW result is
  column-shaped), `norm·gamma` uses `BroadcastDim::Row` (`[1,W]` operand).
  Matches design §10. `EltwiseShape::of(1, BLOCK_SIZE)` ≡ `EltwiseShape::tiles(
  BLOCK_SIZE)` (`{Ht=1,Wt=BLOCK_SIZE,blk=1}`) — equivalent, not a defect.
- **Prompt-rule check (`eval/prompts/rms_norm.txt`).** All applicable MUST rules
  hold: no host-side `to_layout`/`tilize`/`untilize`/`pad`/`slice` in the entry
  point (RM↔tile done natively in the kernels); `default_compute_kernel_config()`
  is the single exported factory (read by `validate()` and the golden tagger);
  output layout matches input; rank<2 and gamma-last-dim-mismatch raise
  `ValueError`; the `"none"` sentinel is accepted for `gamma_dtype`/`gamma_layout`.
  No violations.

### What was fixed

- **Golden observe-shim drift (`eval/golden_tests/rms_norm/axes.py`).** The
  runtime axis classifier `classify_call` omitted the `memory_layout` axis,
  while the op's `SUPPORTED`/`validate()` and the declared cartesian both carry
  it. This made `verify_supported` see "axis 'memory_layout' missing" for the 52
  `test_translated.py` cells and misfile every in-SUPPORTED interleaved cell as
  **`xpass_drift`**. Fixed by adding
  `"memory_layout": input_tensor.memory_config().memory_layout` to `classify_call`
  (mirrors `validate()`). Re-ran golden + CLI: drift → 0, those 52 cells moved to
  `supported_pass` (420 → 472). This is a test-harness consistency fix, not an op
  change; the op was already correct.

### Deferred to refinements (architectural, not fixable in this pass)

None deferred as "known issues." The TARGET−SUPPORTED gap is the refinement
queue (`op_requirements.md`); nothing in the current SUPPORTED rectangle is
broken or degraded.

## Registry Conformance

- **Confirmed present and correctly wired** in `ttnn/ttnn/operations/rms_norm/rms_norm.py`:
  `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()`. `validate()` is the
  first line of the public entry point. Order is per-axis SUPPORTED → cell-level
  EXCLUSIONS. Taggers have the `(inputs, axes)` signature.
- **`INPUT_TAGGERS = {alignment, rank}`** — matches the golden harness. `gamma_dtype`
  / `gamma_layout` / `memory_layout` are read directly off the tensors (pinned
  axes in the cartesian, read via `.dtype`/`.layout`/`.memory_config()`), not
  taggers — consistent with `helpers.py:run_rms_norm` and `axes.py:classify_call`.
- **Op file does NOT declare `INVALID`** — correct under the registry model.
- **`EXCLUSIONS = []` is justified at Phase 0.** The one precision refusal the
  design names, `{float32, fp32_dest_acc_en=False}`, is currently *outside* the
  SUPPORTED rectangle (`fp32_dest_acc_en` supports only `[True]`), so the per-axis
  gate already refuses it. It becomes an explicit EXCLUSIONS entry the moment
  Refinement 1 adds `False` to SUPPORTED (called out in the queue).
- **Auto-fixes to SUPPORTED from XPASS evidence:** none needed. All 52
  `xpass_drift` cells were already inside SUPPORTED; the drift was the observe-shim
  bug above, not an under-claim.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`)

All 16 INVALID entries are well-formed against the three sanity rules — no change
requested:

- **Single-tensor coupling / no cross-tensor-axis entries.** `{bf8b, RM}` couples
  activation dtype×layout; `{gamma_dtype:bf8b, gamma_layout:RM}` couples the gamma
  tensor's own dtype×layout — both same-tensor. The `no_gamma`↔`"none"` sentinel
  couplings and the `gamma present ⇒ real dtype/layout` pairs are
  canonicalization-only (the sanctioned multi-axis exception). The author-scoped
  `{layout:RM, memory_layout:*_SHARDED, gamma_layout:TILE}` triples couple the
  activation's layout+placement with the gamma layout — a documented, deliberately
  parked combination.
- **Canonical `bf8b + ROW_MAJOR` present** for both the activation (`{bf8b, RM}`)
  and gamma (`{gamma_dtype:bf8b, gamma_layout:RM}`) — bf8b is block-quantized, RM
  has no blocks. Structural.
- **No-weight canonicalization present** (norm-like op with a weight axis): the
  five `{no_gamma, gamma_dtype/layout != none}` + two `{gamma, gamma_*: none}`
  entries collapse the gamma-format cartesian to the single `("none","none")` cell.
- **`bf8b + non-aligned`** (`w_non_aligned`, `h_non_aligned`) is parked in INVALID
  as author-scoped "for now" (block-quant + masked/padded reduce out of scope), so
  bf8b is exercised only at `tile_aligned`. This correctly keeps those cells out of
  the refinement backlog; the numeric refinement adds bf8b only at tile_aligned.

## Precision Baseline

Measured at the Phase-0 supported corner (TILE, RM gamma, `fp32_dest_acc_en=True`,
HiFi4), gamma present, via `test_rms_norm_precision_baseline.py`
(`assert_with_pcc` + `comp_allclose`). Ratio = got/true over finite,
|ref|>1e-3 elements (the scale-bug detector).

| Shape | dtype | Max Abs Err | Mean Abs Err | Rel RMS Err | ratio median | ratio p5→p95 |
|-------|-------|-------------|--------------|-------------|--------------|--------------|
| (32,64)        | bf16 | 0.024317 | 0.001576 | 0.003303 | 0.99940 | 0.994→1.005 |
| (2,64,128)     | bf16 | 0.082734 | 0.001730 | 0.003367 | 0.99940 | 0.994→1.005 |
| (2,4,128,512)  | bf16 | 0.088552 | 0.001763 | 0.003304 | 0.99984 | 0.994→1.005 |
| (1,1,128,4096) | bf16 | 0.069851 | 0.001773 | 0.003404 | 1.00085 | 0.995→1.006 |
| (32,64)        | f32  | 0.009038 | 0.000712 | 0.001322 | 0.99881 | 0.997→1.000 |
| (2,64,128)     | f32  | 0.020310 | 0.000821 | 0.001456 | 0.99876 | 0.997→1.000 |
| (2,4,128,512)  | f32  | 0.024617 | 0.000795 | 0.001404 | 0.99887 | 0.998→1.000 |
| (1,1,128,4096) | f32  | 0.014504 | 0.000397 | 0.000767 | 1.00005 | 0.999→1.001 |

**Assessment.** Accuracy is excellent and stable across ranks/shapes/widths.
The got/true ratio clusters tightly around **1.0** (bf16 std ≈ 0.003, f32 std ≈
0.0007) with no drift as W widens — this is ordinary rounding noise, **not** a
scale/structural bug (no tight cluster around a non-1.0 constant). PCC is well
above the per-dtype gates on every cell.

**Recommended tolerances (match the golden `TOLERANCES`):**
- float32: PCC ≥ 0.999, rel-RMS ≤ 0.02
- bfloat16: PCC ≥ 0.995, rel-RMS ≤ 0.04
- (bfloat8_b, once added by Refinement 1: PCC ≥ 0.99, rel-RMS ≤ 0.10)

## Verifier CLI Summary

From `verifier_report.json` (this directory; source run `/tmp/rms_norm_results2`),
after the observe-shim fix:

- supported_pass: **472**
- xfail_expected: **6051**
- invalid_skipped: **33900**
- no_axes_found: **15** (all `test_regression.py` `@numerics` cells — not
  registry-driven; all passed; expected)
- **supported_fail: 0** ✓
- **xpass_drift: 0** ✓
- **xfail_wrong_mode: 0** ✓
- supported_marked_xfail: 0, invalid_unexpected: 0, infeasible_skipped: 0

All three loud categories are clean. Acceptance suite: **70/70**; debug suite:
**9/9**; precision baseline: **16/16**.

### The 6051 `xfail_expected` cells = the TARGET−SUPPORTED gap (all queued)

Breakdown by offending axis value (a cell may be out on >1 axis):

| Offending axis value | cells | Refinement |
|---|---|---|
| `fp32_dest_acc_en = False`   | 3255 | R1 (numeric) |
| `dtype = bfloat8_b`          | 960  | R1 (numeric) |
| `gamma_dtype = bfloat8_b`    | 860  | R1 (numeric) |
| `gamma_layout = TILE`        | 2599 | R2 (tiled gamma) |
| `memory_layout = HEIGHT_SHARDED` | 1501 | R5 (local shard) |
| `memory_layout = WIDTH_SHARDED`  | 1505 | R4 (cross-core) |
| `memory_layout = BLOCK_SHARDED`  | 1502 | R4 (cross-core) |

Every gap value maps onto a queued refinement (or is covered by INVALID). No
`xfail_expected` bucket is left unqueued; no queue gap.

## Recommendations

- **Perf priority / anchor.** The interleaved LLM perf loose cases in
  `feature_spec.LOOSE_CASES` run at `fp32_dest_acc_en=False` + TILE gamma — both
  currently unsupported. R1 (numeric) + R2 (tiled gamma) are ordered first
  precisely to land that config; the first perf pass (R3) then optimizes the
  interleaved **prefill** shapes (rows=8192 fill the grid). The **decode** perf
  shapes (rows=32 → `R=1` tile-row → 1 core) cannot be sped up without the
  cross-core W-split (R4) — they are deferred to the R6 perf pass, not R3.
- **Two-pass DRAM double-read is the main interleaved-perf headroom.** The kernel
  streams each row from DRAM twice (pass-1 Σx², pass-2 normalize). Every perf
  loose-case width (W ≤ 7168 → ≤ 224 tiles ≈ 0.45 MB bf16) fits one core's L1, so
  the design's **lamp-1 resident single-read fast-path** (load the row once, do
  both passes over L1) would roughly halve input DRAM traffic on those cells.
  This is the natural R3 lever (`master.md`: `double_buffer` "keep bytes in
  flight" + the redundant-read-elimination idea). Noted here, exercised in R3.
- **gamma is re-read once per tile-row** (the design's reuse-shared axis at its
  phase-1 value). Holding gamma resident per core / mcasting it once
  (`master.md: shared_input_reuse`, T3) is a later perf lever — folded into the
  perf passes, not a generality gap.
- **L1 pressure is bounded — no OOM risk in the supported set.** Because no CB
  scales with `Wt`, the wide interleaved cells (up to W=8192 in `INPUTS`, and the
  W=16384/32768 loose cases once TILE gamma lands) stream without OOM; they are
  correct-but-single-core, which is a perf/parallelism concern (lamp 2 / R4), not
  a memory one. No `/memory-budget-metal` refinement is warranted for rms_norm.
- **Upstream kernel-lib bug (report, do not fix here).** Two streaming-reduce
  wrapper helpers are stale against the current `reduce<>` /
  `prepare_reduce_scaler<>` template signatures (details in Code Review). They
  are unused by this op (worked around in-kernel) but will bite the next op that
  reaches for `accumulate_reduce<>` or `prepare_partial_reduce_scalers<>`.
  Recommend filing a kernel-lib fix so the wrappers compile again; then this op's
  compute/reader can drop the manual routing and call the wrapper.
- **Observe-shim parity.** `axes.py:classify_call` now includes `memory_layout`;
  keep it in lockstep with `validate()` whenever a new tensor-derived axis is
  added (the shim must mirror the op's axis construction, or translated cells
  drift again).
