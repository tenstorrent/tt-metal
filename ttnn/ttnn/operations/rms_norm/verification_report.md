# Verification Report: rms_norm

## Code Review

### Fixes applied

1. **Added the registry contract** (`SHAPE_TAGGERS` / `SUPPORTED` /
   `EXCLUSIONS` / `validate()`) to `rms_norm.py`. The op file was missing
   the entire block — `eval/golden_tests/rms_norm/test_golden.py` imports
   these symbols from `ttnn.operations.rms_norm`, so golden tests could
   not even be collected before this verification pass. Implemented:
   - `tag_alignment(inputs, axes)` — three-value tagger
     `tile_aligned | w_non_aligned | h_non_aligned` matching the
     feature_spec expectation.
   - `tag_rank(inputs, axes)` — returns `len(inputs[0])`.
   - `tag_shape_size(inputs, axes)` — op-local `small | large` split
     at `Wt ≤ 32` (see "Capability snapshot" below for the rationale).
   - `SUPPORTED` — dtype/layout/alignment/rank/shape_size/gamma_mode/
     gamma_dtype/gamma_layout, sized to what Phase 0 actually supports.
   - `EXCLUSIONS` — TILE_LAYOUT vs non-aligned alignment, gamma_layout
     mismatch, and the mixed-precision-gamma + TILE-input case
     (see "Outstanding kernel issue" below).
   - `validate(input_tensor, gamma)` — raises `ValueError` for pure
     shape sanity (rank<2, gamma-shape mismatch); raises
     `NotImplementedError` for SUPPORTED/EXCLUSIONS misses (so the
     test harness' `xfail(strict, raises=NotImplementedError)` decoration
     is satisfied). Called as the first statement of the public
     `rms_norm()` entry point.
   - `__init__.py` re-exports the four registry symbols + `rms_norm`.

2. **Fixed a descriptor-side off-by-N chunk-count bug**. `num_chunks` was
   `NC * Ht`, which over-counts for RM input with non-tile-aligned `H` and
   `NC > 1` (the reader feeds `NC*H` rows of data but compute iterates
   over `NC*Ht*32` rows' worth of chunks, hanging on the empty trailing
   chunks). Replaced with `num_chunks = ceil(NC*H / 32)`, which is
   identical to `NC*Ht` when `H%32==0` (the TILE path) and shrinks to the
   correct flat-row count for the RM partial-H path. Caught a
   `4×8×47×256` RM-input device hang in the golden suite. See
   `rms_norm_program_descriptor.py:71-82`.

3. **Stage A register-format reconfig**. Added
   `CopyTileReconfig::Input` to the Stage A `CopyTile` element and
   `PackTileReconfig::Output` to the Stage A `PackTile` element in
   `rms_norm_compute.cpp:116-128`. Without these, the unpack/pack
   register format set up by Phase 0's gamma tilize (gamma_dtype)
   persists into Stage A, which then reads `cb_input_tiles` (input_dtype)
   via the wrong unpack format. This was the cause of the catastrophic
   numerical blow-up on the mixed-precision-gamma + TILE-input path
   (probe_001 saw 1000× outliers; after the fix the residual is a
   uniform ~1.27× amplification — still wrong, but no longer catastrophic).
   The full fix for mixed precision is deferred to a refinement (see
   "Outstanding kernel issue" below).

### Outstanding kernel issue (handled via EXCLUSIONS)

Mixed-precision gamma + TILE input (`input_dtype ≠ gamma_dtype`,
specifically `{fp32 input + bf16 gamma}` and `{bf16 input + fp32 gamma}`)
produces a uniform ~1.27× output amplification even after the Stage A
reconfig fix. The chain-driven Stage E reconfig (`InputAndOutput`) does
re-program srcA / srcB / pack for the mismatched gamma case, so the
residual error is most likely the absence of
`UnpackToDestMode::UnpackToDestFp32` on the fp32-side CBs (numerical_stability.md
flag, point 4) — the unpacker quietly truncates fp32 reloads through
SrcA/SrcB at TF32 precision, and the symmetric loss across all elements
compounds into a constant scale offset. Tested cells fail with `severity=bug`
under the golden suite's tolerance.

These 4 axis cells live in `EXCLUSIONS` for Phase 0 and are queued as
**Refinement 3** in `op_requirements.md`. The matched-dtype cells
(`gamma_dtype == input_dtype` on either layout) work correctly and are
in SUPPORTED. Mixed-precision gamma on the RM input path also works
(the in-kernel input tilize re-establishes the correct unpack format),
and is in SUPPORTED.

### Notes for future cleanup (not blockers)

- The compute kernel has explicit `compute_kernel_lib::tilize_config::*`
  enum names spelled out at each call site. Aliasing
  `tilize_config` and `untilize_config` to short names in the kernel's
  anonymous namespace would shrink the call lines significantly. Cosmetic
  only.
- The reader kernel's gamma stick read uses a raw
  `noc_async_read` + barrier rather than a helper. The op design
  explicitly justifies this (`read_sticks_for_tilize` with ROW
  granularity assumes ≥1 page iteration; single-stick gamma is a special
  case). Acceptable.
- `cb_input_tiles` is sized `Wt` (single-buffer). Doubling to `2*Wt`
  would allow the next chunk's read to start while the current chunk is
  in Stage D, but the test suite shows no perf wall here at v1. Out of
  scope.

## Registry Conformance

- **SHAPE_TAGGERS**, **SUPPORTED**, **EXCLUSIONS**, **validate()**: all
  present and wired correctly in `rms_norm.py`. Confirmed by:
  - `python3 -c "from ttnn.operations.rms_norm import SHAPE_TAGGERS,
    SUPPORTED, EXCLUSIONS, validate"` runs clean.
  - `eval/golden_tests/rms_norm/test_golden.py` collects all 2535 cases.
- **Op file does NOT declare INVALID** — confirmed; INVALID lives only in
  `eval/golden_tests/rms_norm/feature_spec.py`.
- **No auto-fixes from XPASS evidence** — the verifier CLI reports 0
  `xpass_drift`, so SUPPORTED is honest.

### INVALID audit (eval/golden_tests/rms_norm/feature_spec.py)

Reviewed the 5 INVALID entries against the three sanity rules:

| Entry | Single-tensor coupling | Universe-must-change | Canonical-redundancy |
|---|---|---|---|
| `{dtype=bf8b, layout=ROW_MAJOR}` | ✓ both axes describe activation | ✓ bf8b is a block format, only TILE meaningful | n/a |
| `{gamma_dtype=bf8b, gamma_layout=ROW_MAJOR}` | ✓ both axes describe gamma | ✓ same as above | n/a |
| `{gamma_mode=no_gamma, gamma_dtype=bfloat16}` | n/a (canonicalization) | n/a | ✓ collapses redundant cells when gamma is None |
| `{gamma_mode=no_gamma, gamma_dtype=bfloat8_b}` | n/a (canonicalization) | n/a | ✓ same |
| `{gamma_mode=no_gamma, gamma_layout=ROW_MAJOR}` | n/a (canonicalization) | n/a | ✓ same |

All five entries are well-formed. **Canonical no_gamma cell** is
`{gamma_dtype=float32, gamma_layout=TILE_LAYOUT}` (the cell left
*outside* INVALID by the three no_gamma entries). SUPPORTED accepts that
cell, and `validate()` fills in those exact canonical values when
`gamma is None`. ✓

The canonical bf8b+ROW_MAJOR activation entry is present (the universe
needs different hardware, not a kernel change), so the audit passes.
No edits to `feature_spec.py` requested.

## Precision Baseline

Measured by `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`
across 4 shapes × 2 dtypes × 2 layouts (16 cases, all pass).

### TILE layout

| Shape | dtype | max_abs | mean_abs | rel_rms |
|-------|-------|--------:|---------:|--------:|
| (1, 1, 32, 32)   | bf16 | 0.0313 | 0.00212 | 0.00558 |
| (1, 1, 64, 128)  | bf16 | 0.0625 | 0.00226 | 0.00513 |
| (2, 1, 128, 256) | bf16 | 0.0625 | 0.00235 | 0.00519 |
| (1, 1, 32, 1024) | bf16 | 0.0938 | 0.00267 | 0.00564 |
| (1, 1, 32, 32)   | fp32 | 0.0137 | 0.00101 | 0.00173 |
| (1, 1, 64, 128)  | fp32 | 0.0177 | 0.00098 | 0.00162 |
| (2, 1, 128, 256) | fp32 | 0.0217 | 0.00088 | 0.00162 |
| (1, 1, 32, 1024) | fp32 | 0.0207 | 0.00093 | 0.00164 |

### ROW_MAJOR layout

| Shape | dtype | max_abs | mean_abs | rel_rms |
|-------|-------|--------:|---------:|--------:|
| (1, 1, 32, 32)   | bf16 | 0.0313 | 0.00212 | 0.00558 |
| (1, 1, 64, 128)  | bf16 | 0.0625 | 0.00226 | 0.00513 |
| (2, 1, 128, 256) | bf16 | 0.0625 | 0.00235 | 0.00519 |
| (1, 1, 32, 1024) | bf16 | 0.0938 | 0.00267 | 0.00564 |
| (1, 1, 32, 32)   | fp32 | 0.0162 | 0.00124 | 0.00207 |
| (1, 1, 64, 128)  | fp32 | 0.0192 | 0.00121 | 0.00195 |
| (2, 1, 128, 256) | fp32 | 0.0270 | 0.00109 | 0.00195 |
| (1, 1, 32, 1024) | fp32 | 0.0232 | 0.00115 | 0.00197 |

**Assessment**: precision is healthy and dtype-aligned.
- bf16 max_abs scales with `1 / sqrt(W)`-ish — single-tile ~0.03,
  wide-W ~0.09. mean_abs flat near `2.3 × 10⁻³` (one bf16 mantissa LSB).
- fp32 max_abs flat near `2 × 10⁻²`; mean_abs `1 × 10⁻³`. The ceiling is
  consistent with the missing `UnpackToDestMode::UnpackToDestFp32` on
  intermediates (numerical_stability.md, point 4) — without that mode
  the fp32 dest still reloads through SrcA/SrcB at TF32 precision.
- TILE and ROW_MAJOR are numerically equivalent on bf16; ROW_MAJOR fp32
  is fractionally worse than TILE fp32, consistent with the extra
  in-kernel tilize op on the RM path.

**Recommended tolerances** (matching golden suite + acceptance test):
- bf16: PCC ≥ 0.995, rtol/atol = 0.05.
- fp32: PCC ≥ 0.999, rtol/atol = 0.005.

## Verifier CLI Summary

(after `eval/eval_test_runner.sh eval/golden_tests/rms_norm/ /tmp/rms_norm_verify`
+ `python3 -m eval.verify_supported /tmp/rms_norm_verify ttnn.operations.rms_norm`)

```
Total tests: 2535

# Expected ✓
  supported_pass:   210
  xfail_expected:   840
  invalid_skipped:  1470

# Signal ✗ (verifier acts on these)
    xpass_drift               0
    supported_fail            0
    xfail_wrong_mode          0
    supported_marked_xfail    0
    invalid_unexpected        0
```

All loud categories are 0. Acceptance suite (`test_rms_norm.py`) reports
**154 / 154** passing. Precision baseline reports **16 / 16** passing.

The 15 "no_axes_found" warnings are `test_regression.py` cases — these
are numerics-only tests not driven by the registry parametrize sweep
(per the conftest's marker registration).

## Subagent Findings (Synthesized)

### From `numerical_stability.md`

- **Single biggest precision lever**: exposing `compute_kernel_config`
  in the entry point. Today, `math_fidelity` is hard-pinned to HiFi4
  and `math_approx_mode` is hard-pinned False. Callers cannot opt into
  HiFi2 for perf or HiFi3 to dodge the Wormhole B0 HiFi4+fp32-dest
  hardware bug (#38306). Mapped to **Refinement 2** in `op_requirements.md`.
- **Missing `UnpackToDestMode::UnpackToDestFp32`** on `cb_x_sq`,
  `cb_mean_sq`, `cb_x_norm`, `cb_output_tiles`. This is what the
  numerical analysis flags as the residual fp32 precision wall. It's
  also the most likely root cause of the mixed-precision-gamma
  TILE-input failure. Bundled into **Refinement 3**.
- **Divide-then-sum** scaler is a minor perf/precision concern (W−1
  extra rounding events vs sum-then-divide). Not a SUPPORTED-axis
  unlocker, so noted here and not queued as a refinement. Would also
  let `cb_scaler` go away entirely — see "Future perf notes" below.
- **Welford's algorithm**: correctly absent (RMSNorm is mean-of-squares,
  not mean+variance). No refinement needed.

### From `data_transfer.md`

- **L1 budget at wide W**: the dominant data-transfer concern. Wt > 32
  (W > 1024) exceeds the per-core 1.5 MB cap even with bf16. This is the
  blocker for `SUPPORTED["shape_size"]=["small", "large"]`. Mapped to
  **Refinement 1** (W-blocking via `accumulate_reduce_block`). Bundles
  naturally with multi-core distribution since wide-W workloads benefit
  most from parallelism.
- **Gamma re-tilization** every chunk is a perf cost (`NC*Ht` redundant
  reads) but doesn't unlock a SUPPORTED cell. Noted; not a refinement.
- **Single-core only**: embarrassingly parallel; bundles into
  Refinement 1.

## Recommendations

The refinement queue (`op_requirements.md`) is intentionally short — 3
refinements — and ordered by ratio of (cells unlocked) / (engineering
work). Beyond those:

1. **Future perf notes** (not refinements; for the kernel implementer or
   a future perf pass):
   - Hoist gamma tilize out of the chunk loop and keep
     `cb_gamma_tiled` persistent. Halves DRAM gamma traffic and removes
     `NC*Ht − 1` tilize calls.
   - Switch reduce scaler from `1/W` (divide-then-sum) to `1.0`
     (sum-then-divide), folding the `* 1/W` into the SFPU
     `add_unary(eps) + rsqrt` stage. Also removes the bf16 `cb_scaler`
     CB entirely.
   - Batch per-tile `noc_async_write_tile` in the TILE writer into a
     fused wait/pop loop.

2. **Stage A reconfig fix that landed in this pass** (point 3 of "Fixes
   applied") is correct and should stay even after the mixed-precision
   refinement lands — it makes the kernel's format flow explicit and
   compile-time-elided when redundant.

3. **Acceptance test gap to flag for the next test author**: the
   immutable `test_rms_norm.py` only exercises matched-dtype gamma
   (gamma is created with the same dtype as input). Mixed-dtype gamma
   coverage lives only in the golden suite. Worth a note in
   `op_requirements.md` Refinement 3 so the refinement implementer adds
   an explicit acceptance-test case after the kernel fix lands.
