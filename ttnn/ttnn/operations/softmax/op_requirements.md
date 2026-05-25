# Operation Requirements: softmax

## Definition
- **Formula**: `output[..., i] = exp(x[..., i] − M) / Σ_k exp(x[..., k] − M)`
  where `M = max(x, dim)` if `numeric_stable=True`, else `M = 0`.
- **PyTorch Reference**:
  ```python
  def pytorch_softmax(x, dim=-1, *, numeric_stable=True):
      original_dtype = x.dtype
      return torch.softmax(x.to(torch.float32), dim=dim).to(original_dtype)
  ```
- **Import Path**: `from ttnn.operations.softmax import softmax`
- **Function Signature**:
  ```python
  softmax(
      input_tensor: ttnn.Tensor,
      dim: int = -1,
      numeric_stable: bool = True,
      *,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED precision**: `["fp32_hifi4_fp32acc"]`
- **SUPPORTED layout**: `[ttnn.TILE_LAYOUT]`
- **SUPPORTED alignment**: `["tile_aligned"]`
- **SUPPORTED rank**: `[4]`
- **SUPPORTED dim**: `[-1, -2]`
- **SUPPORTED numeric_stable**: `[True, False]`
- **EXCLUSIONS**: `[]`
- **Cores**: full Wormhole compute grid via `ttnn.split_work_to_cores`
  (embarrassingly parallel; one reduce-strip per work-item; no inter-core
  communication)
- **Compute config**: HiFi4 + `fp32_dest_acc_en=True` (the only entry in the
  Phase 0 precision name)
- **Golden baseline**: 32 / 40 in-SUPPORTED cells passing; 8 OOM on wide-W
  shapes (W ∈ {4096, 8192}) — moved to Refinement 1.
- **Accuracy**: PCC ≥ 0.9999994 across the 16-cell baseline; max ATOL ≤ 1.06e-3;
  relative RMS ≤ 1.78e-3.

### [x] Refinement 1 — L1 budget fit for wide reduce dimension

**Goal**: rewrite the reduce-dim path so the per-core L1 CB footprint is bounded by
a constant (chunking on the reduce dim) rather than scaling linearly with
`reduce_dim_tiles`. This moves the 8 currently-failing cells (Phase 0 `OOM`,
all on W ∈ {4096, 8192} at `dim=-1`) out of `supported_fail` into passing.
No `SUPPORTED` axis is added — `shape_size` is a resource boundary, not a
kernel-level branch.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: the offending CB is `cb_exps` (between Phase B producer and
Phase C `WaitUpfrontNoPop` consumer — these are sequential helpers, so the CB
must hold a full strip). At `Wt = 128` (W = 4096) that's `128 × 4096 B = 512 KB`
on `cb_exps` alone, plus `2 × Wt × 4096 B = 1 MB` on `cb_input_tiles`, exceeding
the 1.5 MB L1 budget. The skill's `accumulate_reduce_block<>` wrapper handles
Phase A (MAX) and Phase C (SUM) chunking naturally; the trickier piece is
restructuring Phase B (sub+exp) to either (a) recompute exps from a refilled
`cb_input_tiles` chunk in Phase D, or (b) re-stream input through a two-pass
sub-then-exp+mul fused with the multiply by `1/Σ` (online softmax style — but
that's a different algorithm; flag for the implementer to choose the simpler
chunk-and-reload path first). Land this first — it stabilises the resource
envelope before later refinements add new precisions/layouts that further
stress L1.

**Done when**: every Phase 0 cell currently in the `OOM` category passes; the
verifier CLI reports `supported_fail = 0` with the Phase-0 SUPPORTED block
unchanged.

### [ ] Refinement 2 — Numerical configurability (bf16 precisions + compute_kernel_config surface)

**Goal**: add the four bf16 precision modes to `SUPPORTED["precision"]`:
- `bf16_hifi2_fp32acc`
- `bf16_hifi2_bf16acc`
- `bf16_hifi4_fp32acc`
- `bf16_hifi4_bf16acc`

Each name is the (input dtype, math_fidelity, fp32_dest_acc_en) bundle declared
in `eval/golden_tests/softmax/feature_spec.py:PRECISION_CONFIG`. Wire CB format
descriptors and intermediate-CB precision (incl. `UnpackToDestFp32` tagging
where it applies) so all four pass `helpers.TOLERANCES`. Cells that fail out of
the box despite the descriptor wiring (canonically `bf16 + non_tile_aligned_dim`
or `bf8b + non_tile_aligned_dim` if bf8b lands later) go to `EXCLUSIONS`, **not**
their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: the Phase-0 precision-name resolver
(`softmax.py:_resolve_precision_name`) already enumerates the five precision
combos; updating `SUPPORTED["precision"]` to include the four new names is the
mechanical part. The kernel-side work is the CB format propagation
(`cb_input_tiles` / `cb_exps` / `cb_inv_sum` / `cb_output_tiles` become bf16
when the input is bf16) plus checking that the scaler CB (currently fp32 — see
the verification report's "documented deviation" note) still produces correct
SUM/MAX with bf16-typed reduce inputs. Land after Refinement 1 so the L1
budgeting doesn't have to be re-derived for a bf16 envelope first.

**Done when**: all four bf16 precision names appear in `SUPPORTED["precision"]`;
the verifier CLI counts ≥ 32 supported_pass cells per precision (160 total
across precisions) modulo any EXCLUSIONS entries explicitly recorded; `helpers.TOLERANCES`
bands hold without widening.

### [ ] Refinement 3 — Layout (ROW_MAJOR) + rank canonicalization (2D / 3D)

**Goal**: bundle two entry-point shape-handling changes that share the program
descriptor surface:

1. Add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]`. Requires the reader
   path to tilize input data on read (or the program descriptor to insert a
   tilize stage before the existing reader); the writer mirrors with
   untilize-on-write when output layout is ROW_MAJOR.
2. Add `2` and `3` to `SUPPORTED["rank"]`. The kernel and program descriptor
   currently hard-code `n, c, h, w = shape`; canonicalise rank-2 (`H, W`)
   and rank-3 (`B, H, W`) inputs by reshaping to rank-4 with leading `1` dims
   in the entry point (purely a logical view; storage is unchanged).

**Implementation skill**: *(no inventory match — verifier-authored)*

**Verifier notes**: rank canonicalisation is a 3-line entry-point change; the
larger work is the tilize/untilize wrappers for ROW_MAJOR. Both fall on the
program-descriptor / reader / writer surface; bundling them means the
implementer rewrites that surface once. If `xfail_expected` cluster analysis
post-implementation shows ROW_MAJOR + bf16 failing structurally (kernel CB
format mismatch with row-major storage), record those cells in `EXCLUSIONS`
inside this refinement — do not file a separate refinement for them. After
this refinement lands, the only remaining gap is alignment (`w_non_aligned`,
`h_non_aligned`), addressed by Refinement 4. The cross-product of rank ∈ {2, 3}
× layout = ROW_MAJOR pulls in 320 + 320 + 700 = 1340 xfail cells of the original
1360 (some overlap on multi-axis gaps; net new movement here is roughly half the
xfail bucket).

**Done when**: `SUPPORTED["layout"] == [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]`,
`SUPPORTED["rank"] == [2, 3, 4]`, and every cell whose only gap-axes are
`{layout, rank}` passes the golden suite. EXCLUSIONS entries explicitly named
for genuine structural mismatches that surface during implementation.

### [ ] Refinement 4 — Non-tile-aligned shapes (W / H % 32 ≠ 0)

**Goal**: add `w_non_aligned` and `h_non_aligned` to `SUPPORTED["alignment"]`.
Requires partial-scaler handling on the reader side (use
`dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<…>` to emit
a full + partial scaler tile pair) and compute-side wiring of
`ReducePartialScaler::last_tile_at(...)` for both `reduce<MAX>` (Phase A) and
`reduce<SUM>` (Phase C). Padding positions must contribute neutrally: `-∞` for
MAX, `0` for SUM (the `exp(neg_inf − max) = 0` path handles this naturally on
the SUM side once the MAX phase masks correctly).

**Implementation skill**: *(no inventory match — verifier-authored)*

**Verifier notes**: this is the "pad and mask the edge tile" pattern, not a
structural rewrite of the reduction unit (`group_norm`-style straddling
doesn't apply to softmax — the reduce strip is contiguous in either dim).
`reduce_helpers_dataflow.hpp:118–166` documents the partial-scaler API; the
compute side needs `ReducePartialScaler::last_tile_at(reduce_dim_tiles − 1)`
passed to both reduce calls. Phase B (sub+exp) and Phase D (mul) don't need
special handling — the padding columns/rows in `cb_input_tiles` and `cb_exps`
produce numerically correct but unused output (the writer trims via tile-id
math; the existing strip layout already does this for the H/W alignment).
Land last because the partial-scaler dataflow helper is the most localised
piece of L1 budget pressure (one extra scaler tile per CB), and bf16 +
non-aligned could produce ULP rounding that interacts with Refinement 2's
tolerance bands.

**Done when**: `SUPPORTED["alignment"] == ["tile_aligned", "w_non_aligned", "h_non_aligned"]`;
the 600 cells whose only gap is alignment pass the golden suite under the
existing TOLERANCES.
