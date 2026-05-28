# Verification Report: groupnorm_sc_N_1_HW_C

## Code Review

### Fixes applied during this pass

- **Removed `CB_EPS_SCALAR` (slot 11)** — the reader was filling a full scalar
  tile with `eps` at startup, and the compute kernel was `cb_wait_front`-ing on
  it but never consuming the tile (eps actually flows via the CT-arg
  `EPS_BITS_CT` baked into `ckl::AddScalar`). Cost was ~2 KB of L1 and a
  push/wait pair per kernel launch. Slot 11 is left reserved (commented) for
  symmetry with the descriptor's other one-shot scalar CBs.
- **Removed dead-variable cruft** in the compute kernel: `(void)total_iters_g;`
  (variable never read), `(void)EPS_BITS_CT;` (variable *is* read in the very
  next line — the cast was a leftover). Deleted the unused declaration of
  `total_iters_g` itself.

### Deferred / out-of-scope architectural items (not fixable in a verification pass)

- **One-pass variance (`E[X²] − mean²`) is numerically fragile** for two
  regimes that show up in this op:
  1. Large per-group reductions in bf16 (`N_per_g = HW · Cg` ∈ {40960, 81920,
     163840} on SDXL shapes) — the running bf16 accumulator loses precision
     well before reaching the right sum.
  2. Distributions with mean far from zero (e.g. all-positive inputs from
     `test_regression.py::test_distributions[1x1x512x128-positive]`) — `mean²`
     and `E[X²]` cancel almost completely.
  Both are addressable via fp32 destination accumulation + fp32 intermediate
  CBs (Refinement 1) and would benefit further from a switch to Welford or
  two-pass variance (currently out of skill scope; tracked as a follow-up
  inside Refinement 1's verifier notes). I did not rewrite the variance
  algorithm in this pass — the fix that the verifier-report directs at the
  current 18 `supported_fail` cells is the configurability one, not the
  algorithmic one.
- **Manual RM-input zero-fill loop (reader)** — the design suggests
  `dataflow_kernel_lib::read_sticks_for_tilize` for the RM-input path. I kept
  the manual loop because the helper does **not** zero-fill padding bytes
  between sticks, and for `c_non_aligned` (e.g. C=50, last tile has 18 valid
  channels out of 32) the padding lanes' garbage would survive into the tile
  and could be `inf`/`NaN`, poisoning `garbage × 0` in the mask multiply. The
  manual loop pre-zeros the L1 region before the partial-stick reads, which is
  required for correctness on `c_non_aligned`. Switching to the helper would
  require either pre-zeroing the CB write region or adding a `zero_pad=true`
  option to the helper — both architectural changes outside this pass.
- **The 4-way `if constexpr` chain in the apply loop** (HAS_GAMMA × HAS_BETA)
  is verbose but each branch routes the output to the correct CB without
  intermediate copies — a single-helper refactor would re-introduce a
  scratch→output copy and is not a win.
- **`snapshot_tile_to_active_cb` workaround** — the SCALAR-broadcast
  `mul` helper addresses operand B at tile index 0, so to consume tile `g` of
  `cb_group_mean` we copy it to a 1-slot active CB. This is the documented
  helper limitation (called out in the design's compute-kernel doc-comment);
  no change needed here.

## Registry Conformance

- **`INPUT_TAGGERS`** ✓ — `tag_num_groups(inputs, axes)` and
  `tag_alignment(inputs, axes)` both carry the canonical two-arg signature.
- **`SUPPORTED`** ✓ — every axis the kernel gates on (`dtype`, `layout`,
  `alignment`, `affine`, `affine_dtype`, `affine_layout`) is declared. Two of
  the axes (`num_groups`, `alignment`) come from `INPUT_TAGGERS`; `num_groups`
  is intentionally NOT in `SUPPORTED` because it's a continuous per-shape
  scalar — there is no "list of supported num_groups values" the gate can
  meaningfully filter on (and `feature_spec.py` declares `INPUTS` cells with
  `(C/G)%integer == 0` coupling, so the harness never tries an unsupported
  combination).
- **`EXCLUSIONS`** ✓ — empty, as the design indicates.
- **`validate()`** ✓ — runs SUPPORTED gates first, then EXCLUSIONS; raises
  `NotImplementedError` consistently; the public entry point calls
  `validate()` as its first action after the structural `ValueError` checks.
- **No `INVALID` in the op file** ✓ — INVALID lives only in
  `eval/golden_tests/groupnorm_sc_N_1_HW_C/feature_spec.py`.

### INVALID audit (in `feature_spec.py`)

- ✓ `{"dtype": bfloat8_b, "layout": ROW_MAJOR}` — canonical activation bf8b +
  RM impossibility.
- ✓ `{"affine_dtype": bfloat8_b, "affine_layout": ROW_MAJOR}` — same on the
  weight tensors. Both axes describe the same tensor (gamma/beta share these
  axes by spec) so single-tensor-coupling holds.
- ✓ No-weight canonicalization to `(bf16, ROW_MAJOR)` for `affine=no_affine`
  via three entries — matches the norm-like-op-with-weights rule.
- ✓ No cross-tensor-axis entries.
- No issues to flag.

### Auto-fixes from XPASS evidence

None — `xpass_drift = 0`. SUPPORTED matches observed behavior; nothing to
silently widen.

## Precision Baseline

Measured via `tests/.../test_groupnorm_sc_N_1_HW_C_precision_baseline.py`
(bf16 input + TILE input + bf16 gamma/beta + RM gamma/beta — the dominant
Phase 0 SUPPORTED cell).

| Shape | Group cfg | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err |
|---|---|---|---|---|---|
| (1, 1, 32, 32)       | G=1, Cg=32  | 0.999994 | 0.0267 | 0.00357 | 0.0035 |
| (1, 1, 128, 256)     | G=8, Cg=32  | 0.999993 | 0.0809 | 0.00343 | 0.0041 |
| (1, 1, 64, 320)      | G=32, Cg=10 | 0.999991 | 0.0694 | 0.00354 | 0.0043 |
| (1, 1, 1024, 256)    | G=8, Cg=32  | 0.999996 | 0.0952 | 0.00361 | 0.0042 |

**Assessment**: For shapes where the per-group reduction stays inside roughly
`N_per_g ≤ 32768`, the kernel is well within the golden-suite bf16 tolerance
(`pcc ≥ 0.995`, `rel_rms ≤ 0.02`) — PCC is consistently 5-decimal-place tight
to the fp32 reference and rel-RMS is in the 0.0035-0.0043 band. The largest
per-group reduction tested here (`HW · Cg = 1024 · 32 = 32768`) is on the same
order of magnitude as the boundary above which the bf16 single-tile
accumulator visibly degrades — the 18 `supported_fail` cells (Refinement 1)
all live at `N_per_g ∈ {40960, 81920, 163840}`, just past that boundary.

**Recommended tolerances** (carry to refinements as the regression band):
- `PCC >= 0.995` (matches golden bf16 threshold)
- `rel_rms <= 0.02`
- `max_abs <= 0.10`
- `atol = 0.10`, `rtol = 0.05`

## Verifier CLI Summary

Source: `verifier_report.json` in the `<results_dir>/` artifact directory
(too large — ~3.8 MB — to commit to the repo, per the pre-commit large-file
limit; reproduce locally with the runner + verifier-CLI invocation in the
op-level README equivalent below).

Reproduction commands:
```
eval/eval_test_runner.sh eval/golden_tests/groupnorm_sc_N_1_HW_C/ <results_dir>
PYTHONPATH=ttnn/ttnn/operations:$PYTHONPATH \
  python3 -m eval.verify_supported <results_dir> groupnorm_sc_N_1_HW_C \
    --feature-spec eval.golden_tests.groupnorm_sc_N_1_HW_C.feature_spec \
    --output <results_dir>/verifier_report.json
```

| Category | Count | Verdict |
|---|---|---|
| `supported_pass` | 360 | ✓ |
| `xfail_expected` | 3307 | ✓ — bulk of the matrix, gated by SUPPORTED |
| `invalid_skipped` | 3551 | ✓ |
| `supported_fail` | **18** | ✗ → Refinement 1 (numerical-precision on large `N_per_g`) |
| `xpass_drift` | 0 | ✓ |
| `xfail_wrong_mode` | 0 | ✓ |
| `supported_marked_xfail` | 0 | ✓ |
| `invalid_unexpected` | 0 | ✓ |
| `no_axes_found` | 36 | (test_regression.py — not driven by the registry, fine) |

All 18 `supported_fail` cells share the same `failure_category=numerical-precision`
(`severity=precision`, PCC 0.99 — 0.999, RMS 0.03 — 0.22). They group cleanly
by shape: `1x1x4096x320` (Cg=10, HW=4096), `1x1x16384x320` (Cg=10,
HW=16384), `1x1x4096x640` (Cg=20, HW=4096). Per the verifier doc
(`supported_fail` row, OOM/numerical-precision sub-rule), these stay failing
and become a refinement entry; they are NOT moved to EXCLUSIONS because the
metric IS interpretable and the lever (fp32 accumulation, fp32 intermediate
CBs, `compute_kernel_config`) is in scope for the numeric-formats skill.

### `xfail_expected` axis decomposition (input to op_requirements.md)

The 3307 cells outside SUPPORTED group exhaustively into the following
`(axis, missing_value)` pairs from `TARGET[axis] − SUPPORTED[axis]`:

| Axis | TARGET values | SUPPORTED values | Gap |
|---|---|---|---|
| `dtype` | `bf16, fp32, bf8b` | `bf16` | `fp32, bf8b` |
| `layout` | `TILE, ROW_MAJOR` | `TILE, ROW_MAJOR` | — |
| `alignment` | `tile_aligned, hw_non_aligned, c_non_aligned` | `tile_aligned, c_non_aligned` | `hw_non_aligned` |
| `affine` | `gamma_beta, gamma_only, no_affine` | (all three) | — |
| `affine_dtype` | `fp32, bf16, bf8b` | `bf16` | `fp32, bf8b` |
| `affine_layout` | `TILE, ROW_MAJOR` | `ROW_MAJOR` | `TILE` |

Every pair above is covered by a refinement in `op_requirements.md`:

- `dtype ∈ {fp32, bf8b}`           → Refinement 1 (numeric-formats)
- `affine_dtype ∈ {fp32, bf8b}`    → Refinement 1 (numeric-formats)
- `affine_layout = TILE`           → Refinement 2 (layouts)
- `alignment = hw_non_aligned`     → Refinement 3 (alignment)

No `xfail_expected` axis pair is left unaccounted for — the queue covers the
full TARGET surface.

## Recommendations

1. **Refinement ordering matters.** Refinement 1 (numeric-formats + fp32
   accumulator) MUST land before Refinements 2 / 3 because it changes both
   the dtype universe and the intermediate-CB format pipeline. Refinements 2
   and 3 are independent and can land in either order.
2. **The variance algorithm itself (`E[X²] − mean²`) is a numerical
   fragility risk** beyond the dtype/accumulator fix. After Refinement 1
   lands, if `supported_fail` is still non-empty on the largest SDXL shapes
   (HW=16384), a follow-up algorithmic refinement to **two-pass variance**
   (sum-then-recenter) is the next lever. I note this in Refinement 1's
   verifier-notes but do NOT split it into its own refinement up-front —
   the dtype-driven precision lift should be measured first before paying
   the cost of an algorithm rewrite.
3. **`test_regression.py::test_distributions[1x1x512x128-{positive,negative}]`
   are not in the registry** (they live outside the cartesian matrix, so
   `verify_supported` correctly classifies them as `no_axes_found`), but they
   are real test failures that flag the same one-pass-variance instability.
   They give the implementer a check during Refinement 1: if fp32-accumulator
   alone fixes them, the algorithm rewrite can stay deferred; if not, that's
   the trigger for the follow-up two-pass refinement noted above.
4. **No `EXCLUSIONS` are warranted right now.** The 18 `supported_fail` cells
   are not structural capability gaps and not memory-pressure cases — they
   are precision cases with a known lever. Per the verifier doc, those stay
   in the refinement queue, not EXCLUSIONS.
5. **L1 budget headroom is comfortable.** Worst-case Phase-0 L1 footprint is
   ~540 KB per the design's L1 sanity check, well under the 1.5 MB ceiling.
   Refinement 1 (fp32 intermediate CBs) widens the stats CBs by 2× — taking
   them from ~256 KB to ~512 KB worst case — which fits but compresses the
   margin. Worth a budget recheck during Refinement 1; not a current
   blocker.
