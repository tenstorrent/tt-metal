# Verification Report: layer_norm_rm

## Code Review

### Fixes applied during verification

1. **Registry-model conformance** (`layer_norm_rm.py`).  The op file was missing
   the four registry objects (`INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`,
   `validate`).  The Phase-0 implementer left a private `_validate`
   raising `ValueError` for a hand-coded list of structural checks, which is
   neither what `eval/golden_tests/layer_norm_rm/test_golden.py` imports nor
   what `eval.verify_supported` consumes.  Rewrote `layer_norm_rm.py` to declare
   the four objects (taggers `alignment` + `rank`, the SUPPORTED dict described
   below, EXCLUSIONS, and a public `validate()`); kept the existing per-call
   shape invariants as ValueError so the immutable acceptance test continues to
   pass (NotImplementedError extends RuntimeError, so the acceptance test's
   `pytest.raises((ValueError, RuntimeError))` still catches the new gate).

2. **Registry symbol re-export** (`ttnn/operations/layer_norm/__init__.py` and
   `ttnn/operations/layer_norm_rm/__init__.py`).  Both `__init__` modules now
   re-export `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, and `validate` so the
   golden-test harness can do
   `from ttnn.operations.layer_norm import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED`.

3. **RM reader uses the canonical helper** (`kernels/layer_norm_rm_reader.cpp`).
   The Phase-0 implementer hand-rolled a per-row `noc_async_read` loop for the
   RM-input path despite `dataflow_kernel_lib::read_sticks_for_tilize<cb, ROW>`
   being the documented partner of the asymmetric tilize helper on the compute
   side.  Replaced the loop with the helper; behaviour is identical (still one
   stick per `noc_async_read`) but the dependency on the helper contract is now
   explicit, and the file:line reference in `op_design.md` matches the code.

### Items deferred (refinements)

| Item | Why deferred | Refinement |
|---|---|---|
| Replace one-shot `reduce<>` with `accumulate_reduce<>` for streaming | Architectural — changes the inner-loop topology, sizes `cb_input_tiles` independent of `Wt`. Required to unlock W ≥ 4096. | Refinement 2 |
| TILE-layout, multi-batch + non-tile-aligned H (`(4, 8, 47, 256)`) tile-id math | Kernel bug — `tile_id = (start_tile_row + ht) * Wt + wt` assumes a flat tile-row ordering but TTNN pads each leading-dim slice independently. The fix is to either iterate per-(b,c) plane or rebuild tile-id via per-shape strides. Touches the reader/writer. | Refinement 3 |
| fp32 + RM + W non-aligned numerical-precision degradation | Numerical-precision gap in the RM-write path (PCC 0.16–0.80 on W ∈ {47, 100}). Bf16 + RM + same shapes pass; fp32 + TILE + same shapes pass. Currently parked in EXCLUSIONS. | Refinement 3 |
| bfloat8_b activation support | Multiple failures: bf8b output values cause `to_torch` to raise `ValueError: datum for bfp8 is invalid`. Likely a kernel issue producing out-of-range tile values for bf8b's block-shared-exponent format. Not in SUPPORTED. | Refinement 4 |
| Independent gamma/beta dtype vs. input dtype | Op currently rejects `affine_dtype != dtype`. Mixed precision is a known LayerNorm convention (bf16 input + fp32 weights). | Refinement 5 |
| TILE-layout gamma/beta | Op currently rejects affine in TILE layout (the in-kernel tilize step assumes RM sticks). Would unblock a 6th of the golden cells. | Refinement 5 |
| Multi-core distribution over `total_tile_rows` | Embarrassingly parallel (no cross-core dependency). Roll into the streaming-reduce refinement since both touch the program descriptor's per-core slice math. | Refinement 2 |
| HiFi4 is hard-coded for the row reductions (`reduce_helpers_compute.inl:22`) | Numerical-stability finding (see `numerical_stability.md`).  Not in scope to change at the op level; mention only. | n/a |
| `cb_mean` / `cb_inv_std` are sized at input dtype regardless of `fp32_dest_acc_en` | Numerical-stability finding.  Would require descriptor-level changes to force `Float32` and set `unpack_to_dest_mode = UnpackToDestFp32`.  Not in Phase-0 scope. | n/a |

## Registry Conformance

- **Confirmed declared in op file:** `INPUT_TAGGERS` (with taggers
  `tag_alignment`, `tag_rank`), `SUPPORTED` (7 axes), `EXCLUSIONS` (7 entries),
  `validate(input_tensor, gamma=None, beta=None)`.  Each tagger has the
  required `(inputs, axes)` signature.
- **`INVALID` not in op file.**  The op file does not declare `INVALID`; the
  golden suite reads it from `eval/golden_tests/layer_norm_rm/feature_spec.py`
  as required by the registry model.
- **No auto-fixes from XPASS evidence.**  `xpass_drift = 0` in the verifier
  CLI — no SUPPORTED additions needed from observed XPASSes.

### INVALID audit (in `eval/golden_tests/layer_norm_rm/feature_spec.py`)

All five INVALID entries are well-formed per the three sanity rules:

| Entry | Single-tensor coupling? | Universe-must-change? | Notes |
|---|---|---|---|
| `{dtype: bf8b, layout: ROW_MAJOR}` | Yes — both axes describe activation tensor. | Yes — bf8b is a block-quantized format; ROW_MAJOR has no blocks. | Canonical bf8b + RM entry. |
| `{affine_dtype: bf8b, affine_layout: ROW_MAJOR}` | Yes — both axes describe the affine tensor. | Yes — same as above. | Symmetric for gamma/beta. |
| `{affine: no_affine, affine_dtype: bf16}` | Canonicalization — exempt from the same-tensor rule. | n/a (canonicalization). | Canonical no_affine cell is `affine_dtype=float32, affine_layout=TILE`. |
| `{affine: no_affine, affine_dtype: bf8b}` | Same. | Same. | Same. |
| `{affine: no_affine, affine_layout: ROW_MAJOR}` | Same. | Same. | Same. |

No cross-tensor-axis INVALID entries.  No "kernel doesn't support yet"
disguised as INVALID.  No edits to `feature_spec.py` needed.

## Precision Baseline

Captured by `tests/ttnn/unit_tests/operations/layer_norm_rm/test_layer_norm_rm_precision_baseline.py`
(4 shapes × bf16/fp32 in TILE layout with gamma+beta affine, default
`ComputeConfigDescriptor`).

| Shape | dtype | PCC | Max Abs Err | Mean Abs Err | RMS Err | Rel RMS Err |
|-------|-------|-----|-------------|--------------|---------|-------------|
| (1, 1, 32, 64)   | bf16 | 0.999993 | 0.03125 | 0.00260 | 0.00518 | 0.00402 |
| (1, 1, 64, 256)  | bf16 | 0.999993 | 0.12500 | 0.00221 | 0.00512 | 0.00395 |
| (2, 4, 32, 128)  | bf16 | 0.999993 | 0.09375 | 0.00253 | 0.00560 | 0.00408 |
| (1, 1, 32, 100)  | bf16 | 0.999992 | 0.06250 | 0.00280 | 0.00612 | 0.00446 |
| (1, 1, 32, 64)   | fp32 | 0.999992 | 0.05358 | 0.00351 | 0.00563 | 0.00418 |
| (1, 1, 64, 256)  | fp32 | 0.999991 | 0.10558 | 0.00401 | 0.00645 | 0.00440 |
| (2, 4, 32, 128)  | fp32 | 0.999991 | 0.10361 | 0.00434 | 0.00699 | 0.00459 |
| (1, 1, 32, 100)  | fp32 | 0.999989 | 0.06678 | 0.00401 | 0.00630 | 0.00463 |

**Assessment:** PCC ≥ 0.99998 across every shape × dtype.  Mean abs error is
~0.003 (well below the 0.02–0.04 relative-RMS tolerance), max abs error is
0.03–0.12 (consistent with bf16 / HiFi4 + bf16-dest in the broadcast multiplies).
The fp32 numbers are not dramatically tighter than bf16 because **mean,
inv_std, and centered** are all packed back to input dtype between phases —
fp32 input is rounded to fp32 (no loss) but mean/centered remain in fp32 in DEST,
so the bottleneck shifts to the DEST→pack→unpack cycle which is bf16 by default.

**Recommended tolerances** (matches `eval/golden_tests/layer_norm_rm/helpers.py`):
- bf16: PCC ≥ 0.995, rel-RMS ≤ 0.04
- fp32: PCC ≥ 0.9999, rel-RMS ≤ 0.02
- bf8b: PCC ≥ 0.99, rel-RMS ≤ 0.10 (not yet in SUPPORTED — refinement)

## Verifier CLI Summary

`python3 -m eval.verify_supported /tmp/lnrm_golden ttnn.operations.layer_norm
--feature-spec eval.golden_tests.layer_norm_rm.feature_spec
--output ttnn/ttnn/operations/layer_norm_rm/verifier_report.json`

| Category | Count | Notes |
|----------|-------|-------|
| `supported_pass` | 292 | The Phase-0 working envelope. |
| `xfail_expected` | 1535 | Cells outside SUPPORTED, rejected by `validate()` with `NotImplementedError`. |
| `invalid_skipped` | 1855 | INVALID cells from `feature_spec.py`, properly skipped. |
| `supported_fail` | **98** | Real failures. Breakdown below. |
| `xpass_drift` | **0** | Clean: no XPASSes. |
| `xfail_wrong_mode` | **0** | Clean: every xfail-decorated cell raised `NotImplementedError` (not some other exception). |
| `supported_marked_xfail` | 0 | Clean. |
| `no_axes_found` | 15 | The 15 regression tests in `test_regression.py` (they don't carry the `axes` parametrize argument). All passed. |
| **Total** | **3795** | |

### `supported_fail` breakdown (98 cells)

| Category | Count | Cells |
|----------|-------|-------|
| `OOM` | 86 | All shapes with `W ∈ {4096, 8192}`. CB allocation exceeds 1.5 MB L1. Refinement 2 target. |
| `numerical-bug` | 6 | All shape `(4, 8, 47, 256)` × `(TILE layout, h_non_aligned)` × any dtype/affine. Multi-batch leading dims + non-aligned H + TILE input. Refinement 3 target. |
| `numerical-precision` | 6 | All shape `(2, 1, 100, 47)` × `(TILE layout, w_non_aligned)`. Within 3× tolerance; same root cause as the bug above — `prod(shape[:-1]) > 32` with non-aligned H interacts with the tile-id stride math. Refinement 3 target. |

### Hard ship gate

| Gate | Status |
|------|--------|
| `xpass_drift = 0` | ✓ |
| `xfail_wrong_mode = 0` | ✓ |
| `supported_fail = 0` (post-refinement) | ✗ — 98 deferred to Refinement 2 / 3 |

Phase 0 ships with non-zero `supported_fail` because the failing cells fall
into the `OOM` and `numerical-bug` / `numerical-precision` buckets — per the
verifier guide, those legitimately deferred-to-refinement categories don't
gate Phase 0 closure.

## Subagent Findings (Synthesized)

### `numerical_stability.md` (full subagent report)

Top findings, ordered by priority:

1. **HiFi4 is hard-coded for the row reductions** (Pass 1 mean and Pass 2
   variance).  `reduce_helpers_compute.inl:22` forces `HiFi4` on every
   `REDUCE_ROW + SUM/AVG` path regardless of the user's `math_fidelity`
   setting.  The configurable fidelity *does* affect Pass 2's
   `square_in_place`, Pass 3's `mul × inv_std`, and Pass 3's `mul × gamma`.
   This is an architectural choice in the helpers, not a fixable issue at
   the op level.

2. **Reductions use divide-then-sum, not sum-then-divide.**  The scaler
   `1/W` is in the matmul's srcB, so every element is multiplied by `1/W`
   before being summed.  This compounds the rounding pattern by `W` instead
   of once.

3. **`fp32_dest_acc_en=True` is only a partial fix.**  `cb_mean`,
   `cb_inv_std`, `cb_centered` are sized at `input_tensor.dtype` regardless
   of dest-acc mode, and `unpack_to_dest_mode` is not set in the descriptor.
   To get the full fp32 benefit the descriptor would need to (a) force
   those CBs to `Float32` and (b) set
   `compute_kernel_config.unpack_to_dest_mode = UnpackToDestFp32` for those
   CB indices.  Plumbing change, not a kernel rewrite — could be exposed as
   a Refinement 6 if precision becomes a bottleneck.

4. **Catastrophic cancellation in `(x − mean)` is unmitigated.**
   Two-pass algorithm; the `(x − mean)` step happens twice (squared in
   Pass 2, normalized in Pass 3).  No Welford's algorithm because the
   streaming-reduce helpers don't support its per-tile sequential
   accumulation.  Mitigation would be an algorithm-level refinement (out of
   Phase 0 scope).

5. **Non-tile-aligned `W` is correctly handled.**  Partial-scaler pattern
   (`ReducePartialScaler::last_tile_at(1)`) using `origin_W = int(input_shape[-1])`
   (logical width).  No precision regression from W % 32 ≠ 0 on its own.

6. **Epsilon guard is correct.**  `add_unary_tile(dst, eps_bits)` runs
   before `rsqrt_tile`.  Default `epsilon = 1e-5`, configurable, plumbed as
   a fp32 bit pattern through compile-time arg 7.

### `data_transfer.md` (this verifier's analysis)

Top findings:

1. **DRAM bandwidth: 3× input + 1× output.**  The three-pass algorithm
   reads the input tensor three times from DRAM.  Single-core, so no NoC
   balance issue across cores, but the read-three-times design is the
   fundamental Phase-0 bandwidth cost.

2. **L1 footprint scales with `Wt`.**  Hot-path L1 ≈ `(7·Wt + 4) · tile_size`.
   This caps the supported `Wt` at ~96 for bf16 (W ≈ 3072), ~48 for fp32
   (W ≈ 1536).  Observed: `W = 4096` fails with
   "Statically allocated circular buffers grow to 1956352 B which is beyond
   max L1 size of 1572864 B".  This is the **dominant scaling bottleneck**.
   Maps directly to Refinement 2 (streaming reduce removes the `Wt`
   dependency from `cb_input_tiles`).

3. **No inter-core communication.**  Single-core; no semaphores / mcast /
   rings.  Multi-core distribution is trivial (embarrassingly parallel over
   `total_tile_rows`) and folds into Refinement 2 cleanly.

### Cross-doc tradeoffs

- **Precision vs. L1 budget.**  Forcing `cb_mean`/`cb_inv_std`/`cb_centered`
  to Float32 (for true fp32 dest acc benefit) would double their L1
  footprint.  Currently borderline at large `Wt`; would tighten the L1
  budget even more.  Streaming reduce (Refinement 2) decouples this from
  `Wt` and makes the trade affordable.

## Recommendations

- **Refinement priorities in order:** `Refinement 1` (drift fix from
  verifier — currently a no-op since the verifier is clean) skipped.
  `Refinement 2` (streaming reduce + multi-core) unlocks 86 OOM cells and
  is algorithm-fundamental.  `Refinement 3` (TILE-layout multi-batch H tile-id
  math + fp32-RM-W-partial precision) fixes 12 cells and removes 1 EXCLUSION.
  `Refinement 4` (bf8b activation) and `Refinement 5` (independent affine
  dtype/layout) are dtype/layout SUPPORTED expansions.
- **Don't over-invest in the existing one-shot `reduce<>`.**  Most
  Phase-0 follow-up work assumes Refinement 2 has landed.  Resist the
  temptation to fix bf8b *before* the streaming refinement: bf8b's L1
  scaling is even tighter than fp32's, and the wide-W cells will OOM long
  before bf8b is interesting.
- **Code-review notes that did NOT become refinements** (per the verifier
  rule that every refinement must add a SUPPORTED value or move a tracked
  failure into passing):
  - The three-pass DRAM read-three-times pattern.  Possible future
    optimization (cache input in L1 for small `Wt`), but not a SUPPORTED
    addition and no failing cells point at it.  Note only.
  - `cb_mean` / `cb_inv_std` at input dtype.  Numerical-stability finding;
    no `numerical-precision` cells currently fail because of it (the 6
    precision failures are about `(2, 1, 100, 47)` tile-id math).  Note only.
  - HiFi4 hardcoded in the matmul-reduce path.  Architectural; cannot be
    fixed at the op level.  Note only.
