# Operation Requirements: layer_norm_rm

## Definition
- **Formula**:
  ```
  y[..., h, w] = ((x[..., h, w] - mean(x[..., h, :])) /
                  sqrt(var(x[..., h, :]) + epsilon))
                 * (gamma[w] if gamma else 1)
                 + (beta[w]  if beta  else 0)
  ```
  where `mean` and `var` are the population mean/variance over the last
  axis (last-dim normalization).
- **PyTorch Reference**:
  ```python
  def pytorch_layer_norm(x, gamma=None, beta=None, *, epsilon=1e-5):
      x_f = x.to(torch.float32)
      mean = x_f.mean(dim=-1, keepdim=True)
      var = x_f.var(dim=-1, keepdim=True, unbiased=False)
      y = (x_f - mean) / torch.sqrt(var + epsilon)
      if gamma is not None:
          y = y * gamma.reshape(-1).to(torch.float32)
      if beta is not None:
          y = y + beta.reshape(-1).to(torch.float32)
      return y.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.layer_norm_rm import layer_norm`
- **Function Signature**:
  ```python
  def layer_norm(
      input_tensor: ttnn.Tensor,
      gamma: ttnn.Tensor = None,
      beta: ttnn.Tensor = None,
      *,
      epsilon: float = 1e-5,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior
> phases.
>
> **Drift signal**: XPASS-strict failures mean the implementer added
> support but forgot to update SUPPORTED. The implementer fixes by
> updating SUPPORTED.
>
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is
> complete and all tests pass, `[~]` when real work landed but at least
> one named axis value is deferred (treated as completed by the queue,
> surfaced as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED precision**: `["fp32_hifi4_fp32acc"]`
- **SUPPORTED layout**: `[ROW_MAJOR_LAYOUT]`
- **SUPPORTED alignment**: `["tile_aligned"]`
- **SUPPORTED rank**: `[2, 3, 4]`
- **SUPPORTED affine**: `["gamma_beta", "gamma_only", "no_affine"]`
- **SUPPORTED affine_dtype**: `[float32]`
- **SUPPORTED affine_layout**: `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]` — both
  appear because the canonical no_affine cell in `feature_spec.py:INVALID`
  uses TILE; the two `(affine=gamma_*, affine_layout=TILE)` pairs are
  EXCLUSIONS instead.
- **EXCLUSIONS**:
  ```python
  [
      {"affine": "gamma_only", "affine_layout": TILE_LAYOUT},
      {"affine": "gamma_beta", "affine_layout": TILE_LAYOUT},
  ]
  ```
- **Cores**: multi-core (8 × 8 on Wormhole, one strip = 32 RM rows per
  core via `ttnn.split_work_to_cores`)
- **Compute config**: HiFi4 + fp32_dest_acc_en + math_approx_mode=False
- **Golden baseline**: 60 supported_pass, 2635 xfail_expected, 2345
  invalid_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode
  (per `verifier_report.json`).
- **Precision baseline** (see `verification_report.md`): PCC ≥ 0.9999996,
  rel_rms ≤ 1.8e-3, max_abs ≤ 2e-2 across 8 cells.

### [x] Refinement 1 — Numerical configurability expansion (precision + affine_dtype + compute_kernel_config wiring)

**Goal**: add the remaining 3 precision modes to `SUPPORTED["precision"]`
(`bf16_hifi4_fp32acc`, `bf16_hifi4_bf16acc`, `bf8b_hifi4_bf16acc`) and add
`ttnn.bfloat16` and `ttnn.bfloat8_b` to `SUPPORTED["affine_dtype"]`. Wire
`compute_kernel_config.fp32_dest_acc_en` through to the intermediate-CB
format policy (`cb_tilized_x`, `cb_centered`, `cb_mean`, `cb_inv_std`,
`cb_gamma_tilized`, `cb_beta_tilized`) — Float32 when fp32_dest_acc_en is
True, input dtype when False. The scaler CB stays Float32 (precision
deviation, matches softmax-R2's call). Cells that fail out of the box
should land in `EXCLUSIONS`, NOT in their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: should land first — Refinement 2 (layout) and
Refinement 3 (alignment) both reuse the dtype-driven CB format
derivation introduced here. Cells gated on precision alone in the
xfail bucket: 200; on precision + other axes: ~1925 total. Affine_dtype
gated cells: 1470 (predominantly bundled with precision in the same
cartesian face — Refinement 1 unlocks both simultaneously). The softmax
Refinement 2 changelog (`ttnn/ttnn/operations/softmax/changelog.md`) is
the reference precedent: that refinement landed with zero kernel changes
because the kernel was already helper-routed and dtype-agnostic; the
same property holds here — every compute phase goes through
`compute_kernel_lib::tilize`, `binary_op` family, `accumulate_reduce_block`,
`transform_in_place`, `compute_kernel_lib::untilize` — all
dtype-driven via CB format. Expect program-descriptor-only changes.

**Done when**:
- `SUPPORTED["precision"]` = full PRECISION_CONFIG key set (4 modes).
- `SUPPORTED["affine_dtype"]` = `[float32, bfloat16, bfloat8_b]`.
- The intermediate-CB format respects `fp32_dest_acc_en`.
- All 2070 xfail cells whose only gap axes are {precision, affine_dtype}
  pass the golden suite under existing TOLERANCES (no widening). Cells
  that miss the tolerance band move to EXCLUSIONS.

### [ ] Refinement 2 — TILE_LAYOUT input + TILE affine tensors

**Goal**: add `TILE_LAYOUT` to `SUPPORTED["layout"]` (input tensor) and
drop the two EXCLUSIONS entries `{"affine": "gamma_*", "affine_layout":
TILE_LAYOUT}`. After this refinement the op accepts TILE-layout input
(and TILE-layout gamma/beta) end-to-end. Implementation pattern follows
softmax-R3: the kernel beneath this entry point is RM-input / RM-output;
the entry point wraps with `ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)` on
the way in (for TILE input) and `ttnn.to_layout(out, TILE_LAYOUT)` on the
way out (to preserve user-visible layout). Mirror the wrap for gamma /
beta — when supplied with TILE layout, convert to RM before the kernel.

**Verifier notes**: no skill in the current inventory cleanly covers
"accept TILE-layout input by wrapping with `to_layout`" — this is
verifier-authored. Reference the softmax R3 changelog (`ttnn/ttnn/operations/softmax/changelog.md`
§ Refinement 3) for the layout-wrap pattern; the layer_norm_rm version
is the mirror image (softmax accepts RM by wrapping to TILE; this
refinement accepts TILE by wrapping to RM). Cells gated on layout alone
in the xfail bucket: 100; on layout + other axes: ~1540 total. This
refinement is naturally bundled (input layout + affine layout share the
same `to_layout` wrap path); no separate affine-layout refinement.

**Done when**:
- `SUPPORTED["layout"]` = `[ROW_MAJOR_LAYOUT, TILE_LAYOUT]`.
- `EXCLUSIONS` no longer contains the two `(affine=gamma_*, affine_layout=TILE)` pairs.
- Every xfail cell whose only gap-axes are {layout, affine_layout} passes
  the golden suite.
- Output tensor's layout mirrors the input tensor's layout
  (test: round-trip TILE→layer_norm→TILE on a small shape).

### [ ] Refinement 3 — Non-tile-aligned shapes (W / H % 32 ≠ 0)

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to
`SUPPORTED["alignment"]`. The kernel must handle the case where the
W (reduce) dim isn't tile-aligned via the partial-scaler API, and the
H dim non-alignment via ceil-Ht in the program descriptor + strip-level
masking on the last row group. This is the softmax-R4 pattern applied
to a different op family.

**Verifier notes**: no skill in the current inventory cleanly covers
"partial scaler + ceil-Ht for non-tile-aligned shapes". The softmax
Refinement 4 changelog (`ttnn/ttnn/operations/softmax/changelog.md`
§ Refinement 4) describes the exact partial-scaler API and the program
descriptor's ceil-vs-floor division decision. For layer_norm_rm:
- W non-aligned ⇒ reduce dim is W; partial scaler runs in Pass A (mean)
  and Pass B (variance) — both use `accumulate_reduce_block<SUM, REDUCE_ROW>`.
  Use `dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<…,
  partial=W%32>()` on the reader instead of the single-tile
  `prepare_reduce_scaler`, and
  `compute_kernel_lib::ReducePartialScaler::last_tile_at(1)` on each
  `accumulate_reduce_block` (and in Pass C's `sub<COL>` + `mul_in_place<COL>`,
  the broadcast spreads already-valid scalars across the padded W positions
  — no additional masking needed in Pass C).
- H non-aligned ⇒ the strip count is `ceil(H/32)`; the last strip has
  some padded rows that should be masked out on write. This composes
  with the no-affine / affine paths identically; the write is just a
  bounded-stick-count flag on `write_sticks_after_untilize`. Cells
  gated on alignment alone in the xfail bucket: 75; on alignment + other
  axes: ~1155 total.

Cells where bf16 + non_tile_aligned fails the existing TOLERANCES band
become EXCLUSIONS, not separate refinements — same precedent as softmax-R4.

**Implementation skill**: /memory-budget-metal *(secondary — the
streaming-reduce wrappers `accumulate_reduce_block` already documented
in §6 of /memory-budget-metal are the same API that gains the partial
scaler argument; the skill body covers the wrapper's chunking discipline
but the partial-scaler hook itself is described in softmax-R4's
changelog. Treat the skill pointer as a "go read the wrapper rules
first" hint, not a complete methodology for this refinement.)*

**Done when**:
- `SUPPORTED["alignment"]` = `["tile_aligned", "w_non_aligned", "h_non_aligned"]`.
- All 1155 alignment-gated cells pass under existing TOLERANCES (no
  widening). Cells that fail land in EXCLUSIONS (e.g. bf16-acc + tiny-W,
  if any).
- `tag_alignment` is unchanged (already emits the three values).
