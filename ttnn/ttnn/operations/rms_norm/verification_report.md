# Verification Report: rms_norm

Phase 0 verification pass (registry model). Date: 2026-08-04.
Device: Blackhole p150b, 110-core usable compute grid, AICLK ≈ 1.35 GHz.

Artifacts:
* golden results + `verifier_report.json` — `/tmp/rms_norm_verify2/` (copy kept next to this file)
* precision baseline — `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`

---

## Code Review

### Fixed in place

| # | Finding | Fix |
|---|---------|-----|
| 1 | **DRY violation — the block-scoped-CB multiplier was written twice.** `CB_BLOCK_MULT` (which CBs scale with `BLOCK_ROWS · WT_CHUNK`, and at what depth) was spelled out once inside `_resident_fit()` (`depth + 1 + gamma + depth`) and again in the STREAM branch (`cb_x_depth + 1 + gamma + cb_out_depth`). Adding a block-scoped CB or changing a depth would have had to be mirrored in two independent L1 solves — the exact drift the single-source-of-truth rule exists to prevent. | Extracted `_cb_block_mult(depth_x, depth_out, has_gamma)`; both the RESIDENT fit predicate and the STREAM chunk-size solve now call it. `rms_norm_program_descriptor.py`. |
| 2 | **DRY violation — the scaler CB page count restated 3×.** `2 if partial_w else 1` appeared in the L1 budget term and again in the `_cb(CB_SCALER, …)` allocation. | Single `scaler_pages` local; `scaler_bytes` and the CB allocation both derive from it. |
| 3 | **Dead variable `x_resident`** in the descriptor — computed and never used (the kernels derive `X_RESIDENT` from the `NUM_W_CHUNKS` CT arg, correctly). A second, unused copy of a knob is a drift trap. | Removed, replaced with a comment naming `NUM_W_CHUNKS` as the one source. |
| 4 | **Dead knob `GRID_W`** — declared as a primary knob (the design's Lamp L1 handle) but referenced nowhere, so setting it to 2 would have silently produced wrong results (every core computing the whole row as if it owned it). | Kept as a knob and made it live: `create_program_descriptor` now raises `NotImplementedError` for `GRID_W != 1`, naming the missing cross-core combine. |
| 5 | **Bare `except Exception: return []`** in `_cores_in()` — an exception from `ttnn.corerange_to_cores` would have surfaced as a confusing work-split assertion instead of the real error. | Removed the swallow; the row-wise ordering requirement is documented in place. |
| 6 | **Kernel CT-arg offset guard was asymmetric** — the reader asserted its scalar-arg count matches `TensorAccessorArgs<12>()`, the writer had no such guard for `TensorAccessorArgs<8>()`. | Added the matching assert (and moved the reader's before the extend, where it can actually fail early). |

All six are non-behavioural: the golden run before and after is byte-identical
(752 pass / 0 fail, identical verifier summary), and the unit suite is 231/231.

### Verified correct (no change needed)

* **Registry surface** — `INPUT_TAGGERS` (both taggers take `(inputs, axes)`), `SUPPORTED` (every gated axis present, incl. the `"none"` gamma sentinels), `EXCLUSIONS`, `validate()` in the right order (SUPPORTED per-axis → EXCLUSIONS cell-level), `validate()` is the first statement of `rms_norm()`. No `INVALID` symbol in the op file. `PROPERTIES` present and consumed by `eval/dump_op_metadata.py`.
* **Blocking-model fidelity** — `BLOCK_ROWS`, `WT_CHUNK`, `CB_X_DEPTH`/`CB_OUT_DEPTH`, `CB_RM_STAGE_DEPTH`, `REDUCE_BULK`, the L1 budget and the core split are all parameters solved from `L1_SAFETY_FRACTION` + `ttnn.get_max_worker_l1_unreserved_size()`. No CB page count is an unconditional op-dimension: `cb_gamma_tiles`/`cb_gamma_sticks` are the only `Wt`-scaled CBs and they are **predicate-guarded** (counted in the RESIDENT fit, collapsing to `WT_CHUNK` in STREAM) — the sanctioned residency fast-path, not a collapsed knob.
* **Not a half-turned split** — the row axis is split across the *full* grid (count) **and** each core's compute loop walks `BLOCK_ROWS = min(per-core assignment, L1 max)` tile-rows per pass (size). `BLOCK_ROWS = 1` and `WT_CHUNK < Wt` occur only in the STREAM regime, whose predicate *is* the L1-pressure justification.
* **Both dataflow halves batched** — reader and writer share the same `(row-block, width-chunk)` nest and the same `WT_CHUNK`-tiles-per-barrier transaction unit; reader on NoC0 / writer on NoC1 (config defaults). No dribbling half.
* **Helper usage** — every compute phase is a `compute_kernel_lib` helper (`tilize`/`untilize`, `square`, `accumulate_reduce_block`, `mul` ×2, `transform_in_place`); the whole ROW_MAJOR dataflow path uses `read_sticks_for_tilize` / `write_sticks_after_untilize`; scalers use the pool-type-aware `prepare_[partial_]reduce_scaler[s]`. Bulk policies are used where they apply (`WaitPolicy::Upfront` + `OperandKind::Block`, `ReduceInputPolicy::BulkWaitBulkPop`). The only raw LLK is inside the `transform_in_place` lambda — that helper's documented calling convention. No `mcast_pipe` opportunity exists yet (no inter-core communication in Phase 0).
* **CB sync ledger** — push count = wait count per CB in both regimes; `transform_in_place`'s pop-before-reserve on `cb_row_stat` preserves row order in a `BLOCK_ROWS`-page ring (FIFO rotation, verified by hand and by the multi-row tests). Single producer / single consumer per CB per build.
* **APIs** — `TensorAccessor` + `TensorAccessorArgs` (no deprecated `InterleavedAddrGen`), `void kernel_main()`, `api/dataflow/dataflow_api.h` includes.
* **Broadcasts** — `mul<BroadcastDim::Col>` with the column-shaped reduce result as operand **B** (`OperandKind::Col`), `mul<BroadcastDim::Row>` with the row-shaped gamma as operand **B** (`OperandKind::Row`). No CB is filled with replicated tiles: gamma is `Wt` tiles with only row 0 valid, the stat is one tile per tile-row.
* **Padding correctness (the op's headline risk)** — the RMS denominator uses the **logical** `W` (`INV_W = 1/W`, a CT arg) and pad lanes are zeroed out of the reduce by the partial scaler; the ROW_MAJOR staging ring is zeroed once at boot so L1 garbage in pad lanes can't poison the reduce (`inf·0 = NaN`). Confirmed by 24 poisoned-padding acceptance cases + 24 poisoned golden loose cells and by the ratio-spread check in the precision baseline (no uniform scale error).

### Prompt-rule check (`eval/prompts/rms_norm.txt`)

No `## Rules` section. The prose constraints that apply were all checked and hold:
native both layouts with **no** host-side `to_layout` / `tilize` / `untilize` / `pad` / `slice`
(grepped: none in the op or descriptor); non-tile-aligned H and/or W native for both
layouts; one exported `default_compute_kernel_config()` that `None` resolves through and
that the golden tagger imports; the config passed to the compute kernel descriptor
unmodified; `math_fidelity` / `math_approx_mode` not gated; rank<2 and gamma-width
mismatch raise `ValueError` naming "rank"/"gamma"; `"none"` accepted for
`gamma_dtype`/`gamma_layout`.

### Design conformance (`op_design.md`)

Algorithm, phase order, RISC ownership, work split, CB topology and helper mapping all
match §1.5 / §3.3 / §4 / §7. Three deviations, all documented in the descriptor
docstring (D1–D4) and all advisory-level (CB sizing / knob selection, never scheme):

* **D1** `WT_CHUNK` constrained to a *divisor* of `Wt` (uniform chunks, required by
  `tilize`'s compile-time `block_width_tiles`, `BulkWaitBulkPop`'s `pages % cols == 0`
  assert, and the multi-page reserve/ring rule). Still the coarsest admissible value.
  See the perf note on prime `Wt` below.
* **D2** the STREAM solve counts the RM staging CBs at `WT_CHUNK` (what is actually
  allocated) rather than `Wt`. Correct.
* **D3** `accumulate_reduce_block<>` does not expose `reduce<>`'s `ReduceFp32Mode` slot,
  so the fp32 SUM runs at `Fast` instead of the design's `Accurate`. **Deviation from
  §10's checklist.** Measured impact today: none that matters — fp32 output error is
  dominated by a ~2⁻¹⁰ effect elsewhere in the chain (see Precision Baseline), and the
  fp32 golden gate passes with a 13× margin. Carried into Refinement 1's notes, because
  `fp32_dest_acc_en=False` (which R1 adds) makes reduce accumulation the *dominant* term.
* **D4** the RESIDENT predicate *searches* `CB_DEPTH_CANDIDATES` instead of fixing the
  depth. With the shipped `(2,)` this is byte-identical to the design's predicate; the
  search exists so depth stays a live knob. The `(2, 1)` alternative was measured and
  **loses** (0.83× on the one shape it can move) — numbers in the descriptor docstring.

---

## Registry Conformance

* **Confirmed**: `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()` all present and
  correctly wired; `validate()` is called first in the public entry point; the op file
  does **not** declare `INVALID`.
* **Auto-fixes applied to SUPPORTED**: none needed. `xpass_drift = 0` and
  `xfail_wrong_mode = 0` — SUPPORTED is honest, over-claims nothing and under-claims
  nothing.
* **EXCLUSIONS** holds exactly one entry, `{float32, fp32_dest_acc_en: False}`. It is
  currently unreachable (`SUPPORTED["fp32_dest_acc_en"] == [True]`) but documents the
  cell that stays refused once R1 adds `False`. Keep it.

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`)

Well-formed entries (no action):

* `{bfloat8_b, ROW_MAJOR}` — the canonical activation entry, present ✓. Single-tensor.
* `{gamma_dtype: bfloat8_b, gamma_layout: ROW_MAJOR}` — same impossibility on the weight ✓.
* The six `no_gamma` ⇄ `"none"` canonicalization entries, coupled both ways so exactly one
  canonical `("none","none")` cell survives ✓. This is the sanctioned
  canonicalization-of-redundant-cells exception.

Issues to relay (I did not edit `feature_spec.py` — please update via `/golden-tests`):

1. **Three entries are self-declared "author-scoped exclusions", not structural
   impossibilities**, and one of them couples axes across two different tensors:
   `{layout: ROW_MAJOR, memory_layout: *_SHARDED, gamma_layout: TILE}` crosses the
   *activation's* layout+placement with the *weight's* layout. By the single-tensor rule
   and the universe-must-change test these are `EXCLUSIONS` (op-file, revisitable), not
   `INVALID`. Consequence, stated plainly: those 3 × ~n cells are **permanently skipped**,
   so no refinement will ever be filed for them — a documented omission, not a queue gap.
   Same category for `{bfloat8_b, w_non_aligned}` / `{bfloat8_b, h_non_aligned}`
   (single-tensor, so well-formed on that axis, but the comment admits "out of scope for
   now"): they conveniently pre-exclude bf8b's hardest corner from Refinement 1, which is
   the right *engineering* call but the wrong *category*.
2. **Missing entry, by the file's own analogy** (also raised in `op_design.md` §9.2):
   `{gamma_dtype: bfloat8_b, alignment: w_non_aligned}` and `{…, h_non_aligned}`. A bf8b
   gamma shares one exponent per block, so a non-aligned `W` puts gamma's pad lanes in the
   same quantization block as real weights. This is the identical impossibility already
   accepted for the activation. Until it is added, Refinement 1 will have to park those
   cells in `EXCLUSIONS` instead.

---

## Precision Baseline

`test_rms_norm_precision_baseline.py`, 4 shapes × 2 dtypes × 2 layouts,
gamma on, `HiFi4 + fp32_dest_acc_en=True + math_approx_mode=False` (the Phase 0 corner).
TILE rows shown; ROW_MAJOR is identical to 3–4 significant figures.

| Shape | dtype | PCC | Max abs err | Mean abs err | Rel RMS err | ULP max | ratio median (p5…p95) |
|-------|-------|-----|-------------|--------------|-------------|---------|------------------------|
| (1,1,32,64) | bf16 | 0.9999972 | 0.02398 | 0.001214 | 0.00237 | 1 | 0.99964 (0.99567…1.00346) |
| (1,1,128,512) | bf16 | 0.9999972 | 0.03960 | 0.001278 | 0.00241 | 2 | 0.99962 (0.99578…1.00363) |
| (2,1,64,4096) † | bf16 | 0.9999972 | 0.04519 | 0.001233 | 0.00240 | 2 | 0.99968 (0.99577…1.00365) |
| (1,1,50,200) ‡ | bf16 | 0.9999974 | 0.02914 | 0.001189 | 0.00231 | 1 | 0.99973 (0.99586…1.00366) |
| (1,1,32,64) | fp32 | 0.9999997 | 0.01345 | 0.000809 | 0.00152 | 51710 | 0.99873 (0.99743…0.99985) |
| (1,1,128,512) | fp32 | 0.9999997 | 0.02243 | 0.000817 | 0.00148 | 63879 | 0.99877 (0.99746…0.99986) |
| (2,1,64,4096) † | fp32 | 0.9999997 | 0.02457 | 0.000670 | 0.00124 | 66333 | 0.99904 (0.99771…1.00022) |
| (1,1,50,200) ‡ | fp32 | 0.9999997 | 0.01557 | 0.000755 | 0.00151 | 55878 | 0.99870 (0.99741…0.99981) |

† STREAM regime (width-chunked, x re-read).  ‡ both H and W non-tile-aligned.

**Assessment.** Accuracy is uniform across shape, regime, layout and alignment — the
STREAM/chunk-accumulate path and the masked-reduce path cost nothing measurable, which is
the important negative result. bf16 sits at 1–2 ULP of its own grid (rel RMS ≈ 0.0024,
i.e. bf16's 2⁻⁹ step); the golden bf16 gate (PCC 0.995 / RMS 0.04) has ~17× headroom.

fp32 is the interesting row: PCC 0.9999997 and rel RMS 0.0015, but that is ≈ 2⁻¹⁰
*relative* — roughly tf32-grade, ~10⁴ ULP in the fp32 grid, four orders worse than fp32
could deliver. **Scale-vs-precision triage: this is NOT a scale bug.** The got/true ratio
spans 0.9974…0.9999 (a 2.4 × 10⁻³ spread) around a median 1.2 × 10⁻³ below 1 — i.e. the
spread is *wider* than the offset, so the error is distributed, not a constant multiplier;
and rel RMS is 0.0015, nowhere near the ≳0.1 that flags a uniform scale/structural error.
A dedicated probe (constant input, so the stat is exactly representable) pins the residue
at exactly `1 − 2⁻¹⁰`, independent of `W` and of magnitude, and it *shrinks* under
`math_approx_mode=True` (rel RMS 1.5e-3 → 0.9e-3) — so it is an SFPU/FPU datapath
mantissa effect in the `rsqrt`+multiply chain, not accumulation and not the op's
blocking. The precision-baseline test now carries an explicit uniform-scale assertion
(tight ratio cluster ⇒ median must be within 1% of 1.0) so a future padding/scaler
regression cannot hide behind a high PCC.

**Recommended tolerances** (2× margin over measured, both layouts, all regimes):
bf16 — `PCC ≥ 0.9999`, rel RMS ≤ 0.005, `atol = 0.06`.
fp32 — `PCC ≥ 0.99999`, rel RMS ≤ 0.004, `atol = 0.04` (do **not** ask fp32 for
`rtol ≤ 1e-4` on this datapath). The golden suite's current `TOLERANCES` and the perf
loose cases' soft `pcc_threshold = 0.9995` are all comfortably met **at
`fp32_dest_acc_en=True`**; the `False` corner is unmeasured (see Refinement 1 notes).

---

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/rms_norm/` → `PASSED=752 FAILED=0 ERRORS=0
SKIPPED=33902 HANGS=0 TOTAL=40828`, then `python3 -m eval.verify_supported`:

| Category | Count |
|---|---|
| supported_pass | **737** |
| xfail_expected | 6172 |
| invalid_skipped | 33900 |
| infeasible_skipped | 2 |
| **supported_fail** | **0** ✓ |
| **xpass_drift** | **0** ✓ |
| **xfail_wrong_mode** | **0** ✓ |
| supported_marked_xfail | 0 ✓ |
| invalid_unexpected | 2 (see below) |
| no_axes_found | 15 (`test_regression.py`, not registry-driven — all 15 passed) |

All three loud categories are clean, before and after the code-review fixes (the run was
repeated after them; the summary is identical).

`invalid_unexpected = 2` is a harness artifact, not an op signal:
`test_translated.py::test_rms_norm_sharded_uneven_multicore_logical_width[w{72,200}_c*_nonaligned-bfloat8_b]`
match the INVALID cell `{bfloat8_b, w_non_aligned}` but the translated suite decorates
from `is_supported` (xfail) rather than skipping INVALID. Both *did* xfail with
`NotImplementedError` from `validate()`, so op behaviour is correct; only the category
label is wrong. No op-side action.

**xfail_expected accounting** (a non-empty bucket must map onto the queue). Every one of
the 6172 xfails is explained by exactly three missing axis values, and all three are in
the queue:

| Missing axis value(s) | xfail cells | Refinement |
|---|---|---|
| `fp32_dest_acc_en = False` (alone or crossed) | 3607 | R1 |
| `dtype`/`gamma_dtype = bfloat8_b` (alone or crossed) | 1662 | R1 |
| `memory_layout ∈ {HEIGHT,WIDTH,BLOCK}_SHARDED` (alone or crossed) | 4834 | R2 |

(Rows overlap because a cell can miss two axes; the union is 6172, and no xfail cell
misses an axis value outside `{fp32_dest_acc_en=False, bfloat8_b, *_SHARDED}`.) Loose
cases specifically: 382 xfail / 3 pass today — R1 alone converts 100 of them (every
interleaved resilience + pad-poison + interleaved perf case), R2 the remaining 279.

---

## Recommendations

Refinement priorities and the cross-cutting notes live in `op_requirements.md`. What
belongs here instead:

1. **`GRID_W = 1` leaves the grid idle on decode-shaped inputs — by design, and it is the
   single largest perf item.** `Rt = 1` for every `(1,1,32,W)` shape, so
   `split_work_to_cores` hands the whole op to **one** core; wide `W` additionally lands in
   STREAM, which re-reads x (≈2× DRAM bytes). Structurally, `(1,1,32,7168)` bf16 moves
   ≈0.9 MB through one core at the ~13 GB/s single-core ceiling ⇒ tens of µs, against the
   feature-spec reference of 104 µs *with a ≥7× requirement* (≤ 14.9 µs). No knob closes
   that; only the cross-core width combine does. This is design-conformant (`op_design.md`
   R9 / Lamp L1), so it is not filed as a bug — it is the reason R2 is the cross-core
   scheme-change and why the perf phases follow it.
2. **STREAM chunk granularity collapses on prime `Wt`.** D1 forces `WT_CHUNK | Wt`, so a
   prime `Wt` past the residency threshold (e.g. `Wt = 127` for `W = 4064`, a shape in
   `_RESILIENCE_SHAPES`) drops to `WT_CHUNK = 1` — 127 chunks × 2 passes, one tile per
   NoC barrier, i.e. the master.md granularity floor, and gamma re-read per chunk. It is
   *correct* (and only reachable in the L1-fallback regime), but it is the worst
   dataflow shape the op can produce. The clean fix is a ragged tail (a runtime-`wt_c`
   tilize/untilize path plus a non-`% cols == 0` reduce policy) or Lamp L5; noted as a
   perf lever in Refinement 4 rather than as a correctness item.
3. **L1 / memory-pressure observations that do not yet OOM.** No `OOM` cell exists in the
   golden run (`supported_fail = 0`) — the RESIDENT/STREAM predicate keeps the CB
   footprint under `0.85 × get_max_worker_l1_unreserved_size()` for every shape in
   `INPUTS` and `LOOSE_CASES`, including `W = 32768`. Two thin spots to watch when a
   refinement changes CB sizing: (a) `cb_gamma_tiles` (+ `cb_gamma_sticks` for RM gamma)
   is the only `Wt`-scaled allocation and is charged to the *fixed* term, so a wide-`W`
   **fp32 TILE gamma** consumes ~4 KB/tile before any block is sized — at `Wt ≈ 300` that
   is 1.2 MB of the budget on its own and pushes the chunk solve toward `WT_CHUNK = 1`;
   (b) `L1_SAFETY_FRACTION = 0.85` is the one hand-set number and the *only* lever if a
   future refinement adds a CB — lower it there rather than trimming a block factor.
4. **Deferred, architectural (not filed as refinements).** (i) `ReduceFp32Mode::Accurate`
   for fp32 (D3) requires either a helper change in `streaming_reduce_helpers.hpp` or
   dropping to a raw `reduce<>` + manual last-chunk partial-scaler routing; today it buys
   nothing measurable, so it is carried as a note on R1 rather than as work. (ii) The
   `1 − 2⁻¹⁰` fp32 datapath residue (§ Precision Baseline) is below every gate and is not
   specific to this op; if a caller ever needs true fp32, the lever is the rsqrt
   formulation (`sqrt` + `recip`, or an extra Newton step), not the blocking.
5. **Non-finite values in *input* tile padding are not defended.** The masked reduce
   multiplies pad lanes by a zero scaler, so `inf`/`NaN` padding on a TILE input would
   poison the row (`inf · 0 = NaN`). The ROW_MAJOR staging ring is boot-zeroed precisely
   for this reason; the TILE path trusts the tensor's padding to be finite (it is:
   `from_torch` zeroes it, and the poison tests use 1000.0). A mask-based reduce (Lamp
   L6a) would have the same exposure, so this is a note, not a fix.
