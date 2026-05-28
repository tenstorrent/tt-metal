# Operation Requirements: groupnorm_sc_N_1_HW_C

## Definition

- **Formula**: For each group `g` with channels `c ∈ g`:
  `mean_g = (1 / N_per_g) · Σ_{h, c∈g} x[h, c]`
  `var_g  = (1 / N_per_g) · Σ_{h, c∈g} x[h, c]² − mean_g²`
  `y[h, c] = (x[h, c] − mean_g(c)) · rsqrt(var_g(c) + eps) · γ[c] + β[c]`
  where `g(c)` is the group containing channel `c` and
  `N_per_g = HW · (C / num_groups)`.
- **PyTorch Reference**: `torch.nn.functional.group_norm`, called on
  `(N, C, HW)` (we permute `(N, 1, HW, C)` → `(N, C, HW)` for the reference
  and back for the output).
- **Import Path**: `from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C`
- **Function Signature**:
  ```python
  def groupnorm_sc_N_1_HW_C(
      input_tensor: ttnn.Tensor,
      num_groups: int,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      beta: Optional[ttnn.Tensor] = None,
      eps: float = 1e-5,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior
> phases (acceptance + precision-baseline + golden suite).
> **Drift signal**: XPASS-strict failures mean the implementer added support
> but forgot to update SUPPORTED. The implementer fixes by updating
> SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is
> complete and all tests pass, `[~]` when real work landed but at least one
> named axis value is deferred (treated as completed by the queue, surfaced
> as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[ttnn.bfloat16]`
- **SUPPORTED layout**: `[ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]`
- **SUPPORTED alignment**: `[tile_aligned, c_non_aligned]`
- **SUPPORTED affine**: `[gamma_beta, gamma_only, no_affine]`
- **SUPPORTED affine_dtype**: `[ttnn.bfloat16]`
- **SUPPORTED affine_layout**: `[ttnn.ROW_MAJOR_LAYOUT]`
- **Cores**: single (`CoreCoord(0, 0)`, per design `SINGLE-CORE` constraint)
- **Compute config**: hard-coded — default `ComputeConfigDescriptor()` (no
  `compute_kernel_config` is exposed yet). bf16 throughout the dataflow,
  bf16 intermediate CBs.
- **Golden baseline**: 360 / 378 SUPPORTED cells passing (per
  `verifier_report.json`). 18 cells in `supported_fail` (numerical-precision
  on large `N_per_g`) — Refinement 1.
- **Precision baseline** (PCC vs torch.nn.functional.group_norm, fp32 ref;
  bf16 input, bf16 weights, TILE input, ROW_MAJOR weights):
  - (1,1,32,32)    G=1   : PCC=0.999994, rel_rms=0.0035, max_abs=0.0267
  - (1,1,128,256)  G=8   : PCC=0.999993, rel_rms=0.0041, max_abs=0.0809
  - (1,1,64,320)   G=32  : PCC=0.999991, rel_rms=0.0043, max_abs=0.0694
  - (1,1,1024,256) G=8   : PCC=0.999996, rel_rms=0.0042, max_abs=0.0952

### [ ] Refinement 1 — Numerical configurability + precision fix

**Goal**: add `ttnn.float32` and `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and
`SUPPORTED["affine_dtype"]`; expose `compute_kernel_config: ttnn.ComputeKernelConfig`
on the entry point (`math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`,
`dst_full_sync_en`); switch the variance / stats intermediate CBs
(`cb_running_acc_sum`, `cb_running_acc_sumsq`, `cb_group_mean`,
`cb_group_rcp_std`, `cb_scratch_a`, `cb_scratch_b`, `cb_inv_N_scalar`) to
`Float32` format and turn on `fp32_dest_acc_en` so the running accumulator
holds the full reduction in fp32 even when the I/O dtype is bf16. Tag
`UnpackToDestFp32` on the input CBs that feed the SFPU rsqrt/AddScalar chain.
Cells that fail out-of-the-box (most commonly `bfloat8_b + non_tile_aligned_dim`
on activations or weights — i.e. the `c_non_aligned` × `bfloat8_b` cases that
the canonical INVALID `bf8b + ROW_MAJOR` already neutralizes for the RM
combinations) land in `EXCLUSIONS`, not their own refinement.

Side effect (this is the main precision win): the 18 `supported_fail`
cells — `1x1x4096x320`, `1x1x16384x320`, `1x1x4096x640` at `num_groups=32`,
all `bf16` — pass once the running accumulator no longer truncates to bf16
mid-reduction.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: must land first — Refinements 2 and 3 reuse the
dtype-driven CB-format derivation introduced here (the program descriptor
will become dtype-aware after this refinement, not before). After this
refinement, re-run `test_regression.py::test_distributions[1x1x512x128-{positive,negative}]`:
if those still fail (PCC ~0.97 baseline), the residual fragility is from
the one-pass `E[X²] − mean²` formula, not from accumulator precision. In
that case file a follow-up algorithmic refinement to switch the variance
recipe to two-pass (sum-then-recenter): reduce → mean → re-stream input →
`Σ (x − mean)²` → divide. The two-pass refinement is verifier-authored
(no skill currently covers algorithmic numerical-stability rewrites in
detail) and should reference `numerical_stability_analysis_reference.md`
§3.2 and §4.2 for the helper recipes the implementer will need. Do NOT
file it preemptively — measure with the cheap fix first.

**Done when**:
- All 18 currently-failing cells in `verifier_report.json::supported_fail`
  pass (`supported_fail == 0`).
- `SUPPORTED["dtype"]` is `[bf16, fp32, bf8b]` and `SUPPORTED["affine_dtype"]`
  is `[bf16, fp32, bf8b]`.
- `compute_kernel_config` is an optional kwarg on the public entry point.
- Precision baseline (the four shapes in
  `test_groupnorm_sc_N_1_HW_C_precision_baseline.py`) does not regress
  (PCC stays ≥ 0.9999 on those shapes).

### [ ] Refinement 2 — Affine layout TILE

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["affine_layout"]`. Today the
reader expects gamma/beta as row-major sticks (shape `(1,1,1,C)`,
ROW_MAJOR) and replicates each stick 32× in L1 before the compute kernel
tilizes that block. For TILE-laid weights, the gamma/beta tensor is one
tile per `Ct` (zero-padded in rows 1-31 since the logical shape is
`(1,1,1,C)`). The kernel must:
1. Read tile-laid gamma/beta directly from DRAM into `cb_gamma_tile_T` /
   `cb_beta_tile_T` (no replicate-32 staging).
2. Use a row-broadcast (`BroadcastDim::ROW`) `mul` / `add` against the
   per-T tile so the single valid row of the weight tile is broadcast
   down to all rows of the input block. (Alternative: pre-replicate row 0
   across all 32 rows in compute with a single `copy_tile` + broadcast
   write — either pattern works; the current `NONE`-broadcast path
   expects a row-replicated weight tile.)

**Implementation skill**: /memory-layouts

**Verifier notes**: depends on Refinement 1 because the dtype universe
expansion in R1 makes affine weights potentially fp32/bf8b — and those
combinations are gated by the per-tensor `(dtype, layout)` impossibility
table in feature_spec.py's INVALID. The canonical `affine_dtype=bf8b +
affine_layout=ROW_MAJOR` is already INVALID; on the TILE side, all three
affine_dtypes (bf16, fp32, bf8b) are legal and must work. Order this
*after* R1 so the implementer has the descriptor-level dtype derivation
in hand before adding a second layout path. Bundles naturally with R3 if
the implementer prefers a single layout-themed refinement, but the work
units are independent enough to land separately.

**Done when**:
- `SUPPORTED["affine_layout"]` contains `ttnn.TILE_LAYOUT` and
  `ttnn.ROW_MAJOR_LAYOUT`.
- Every previously-`xfail_expected` cell with `affine_layout=TILE` now
  passes the golden suite (in conjunction with whichever
  `affine_dtype` × `dtype` × `alignment` cell it sits in, modulo INVALID).

### [ ] Refinement 3 — Alignment hw_non_aligned

**Goal**: add `hw_non_aligned` to `SUPPORTED["alignment"]`. Today the
design enforces `HW % 32 == 0` and `Ht = HW / 32` (integer); the reader and
writer iterate exactly `Ht` HW-tile-rows per `T`. For `hw_non_aligned`,
the last HW-tile-row covers `HW % 32` valid rows out of 32 and needs
last-tile-row masking in both the reduce phase (the reduction must NOT
count the padding rows; the partial reduce scaler `1 / N_per_g` must use
`N_per_g = HW · Cg` with the true HW, not `Ht · 32 · Cg`) and the apply
phase (the writer must not write the padding rows back to DRAM).
The change is local to:
1. **Reader**: read the partial last HW-tile-row, zero-fill the padding
   rows in L1 before pushing.
2. **Compute**: no algorithm change — the zero-filled padding rows
   contribute zero to the reductions, and `1 / N_per_g` already uses the
   true `HW · Cg` (computed host-side, sent as `INV_N`).
3. **Writer**: track per-T-tile valid row count for the last block and
   write only valid rows to DRAM (this is the only output-shape-correctness
   change).

This is a kernel-internal change — **do not** wrap the op in
`ttnn.to_layout` / pad-then-trim. The `/memory-layouts` skill calls out
the in-kernel pattern (last-tile zero-pad in the reader; valid-row count
in the writer).

**Implementation skill**: /memory-layouts

**Verifier notes**: independent of R2 (different code paths — alignment
touches HW-tile-row iteration; affine_layout touches the per-T weight
read). Either can land first after R1; can also be bundled with R2 if
the implementer prefers a single layout-themed refinement, but keeping
them separate makes the per-axis SUPPORTED change cleaner. **L1 watch:**
this refinement does NOT widen any CB — the reader's RM-input path
already zero-fills the L1 staging region for the partial-C case (same
code path); this refinement reuses that for the partial-HW case.

**Done when**:
- `SUPPORTED["alignment"]` contains `tile_aligned`, `c_non_aligned`, and
  `hw_non_aligned`.
- All `hw_non_aligned` cells in `verifier_report.json::xfail_expected`
  move to `supported_pass`.
- Output shape correctness verified: `y_tt.shape == input.shape` even
  when `HW % 32 != 0`.
