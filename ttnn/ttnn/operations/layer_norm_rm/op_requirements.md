# Operation Requirements: layer_norm_rm

## Definition
- **Formula**: `y = ((x - mean(x, dim=-1, keepdim=True)) / sqrt(var(x, dim=-1, keepdim=True) + epsilon))`; then optional `y *= gamma; y += beta`
- **PyTorch Reference**: `torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],), weight=g, bias=b, eps=epsilon)`
- **Import Path**: `from ttnn.operations.layer_norm_rm import layer_norm_rm` (also re-exported as `ttnn.operations.layer_norm.layer_norm`)
- **Function Signature**:
  ```python
  layer_norm_rm(
      input_tensor: ttnn.Tensor,
      gamma: ttnn.Tensor | None = None,
      beta:  ttnn.Tensor | None = None,
      *,
      epsilon: float = 1e-5,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[ttnn.float32]`
- **SUPPORTED layout**: `[ttnn.ROW_MAJOR_LAYOUT]`
- **SUPPORTED alignment**: `["tile_aligned"]`
- **SUPPORTED rank**: `[2, 3, 4]`
- **SUPPORTED affine**: `["gamma_beta", "gamma_only", "no_affine"]`
- **SUPPORTED affine_dtype**: `[ttnn.float32]`
- **SUPPORTED affine_layout**: `[ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]`
- **EXCLUSIONS**: `[{"affine":"gamma_only","affine_layout":TILE}, {"affine":"gamma_beta","affine_layout":TILE}]`
- **Cores**: full Wormhole grid via `ttnn.split_work_to_cores` (embarrassingly parallel; no inter-core)
- **Compute config**: hard-coded `HiFi4 + fp32_dest_acc_en=True + math_approx_mode=False`
- **Golden suite at Phase 0**: 34 / 1924 supported cells pass; 26 supported cells fail with `OOM` (queued in Refinement 3). 1865 xfail (TARGET-vs-SUPPORTED gap). 1855 INVALID skipped.

### [ ] Refinement 1 — Numerical configurability expansion

**Goal**: add `ttnn.bfloat16`, `ttnn.float32`, `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` for the input tensor; mirror the addition on `SUPPORTED["affine_dtype"]` for gamma/beta. Expose `compute_kernel_config: ttnn.ComputeKernelConfig` on `layer_norm_rm` (defaulting to the current HiFi4 + fp32_dest_acc bundle). Correct intermediate-CB precision (incl. `UnpackToDestFp32` tagging) so the fused tilize → reduce → sub → square → reduce → mul chain doesn't truncate fp32 partials at every helper boundary. Cells that fail out of the box (typically `bfloat8_b + non_tile_aligned_dim` on the input or affine path) land in `EXCLUSIONS`, not in their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: should land first — Refinement 2 (TILE layout) reuses the dtype-driven CB format derivation introduced here. The `cb_scaler` is currently fp32 (advisory deviation from design's bf16); the skill should re-evaluate whether bf16 scaler is appropriate for a bfloat16 input path. INVALID already excludes `(bfloat8_b, ROW_MAJOR)` for both the activation and affine tensors, so the bf8b cells you actually need to handle are TILE-only.

**Done when**: every TARGET cell with `dtype ∈ {bfloat16, bfloat8_b}` or `affine_dtype ∈ {bfloat16, bfloat8_b}` either passes or is in `EXCLUSIONS` with a documented `non_tile_aligned_dim` mask.

### [ ] Refinement 2 — TILE input + TILE affine support

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["layout"]` for the input tensor and **remove both EXCLUSIONS entries** so that `(gamma_only, TILE)` and `(gamma_beta, TILE)` pass. When the input is already tiled, bypass the in-kernel tilize and untilize phases — the input goes directly into `cb_input_tiles` from the reader, and the output goes directly from `cb_norm` to the writer. Same logic for tiled gamma/beta: skip the replicate-32× reader pattern and the boot-time tilize. Internally this is a CT-flag choice on the reader/compute kernels (`layout_is_tile`) and a different program-descriptor wiring for the CBs.

**Implementation skill**: (none in inventory — verifier-authored)

**Verifier notes**: layout work on the input and the affine tensors share the reader rewrite (both currently use the RM-stick replicate path) — keep them bundled. Refinement 1 (dtypes) should land first because the tilize-bypass path needs to honor whatever CB format derivation the dtype work introduced. The W-axis chunking in Refinement 3 also touches the tilize/reduce path, but is independent of layout — keep them separate. Once this lands, all `layout=TILE` and `affine_layout=TILE` cells fall inside SUPPORTED with no EXCLUSIONS.

**Done when**: every TARGET cell with `layout=ttnn.TILE_LAYOUT` (input) or `affine_layout=ttnn.TILE_LAYOUT` (gamma/beta) passes.

### [ ] Refinement 3 — L1 budget fit for wide reduce dim

**Goal**: rewrite the reduction phase so the per-core L1 CB footprint is bounded by a constant (chunking on the reduce dim), so the op stops OOMing on the wide-hidden shapes in `feature_spec.INPUTS` (W ∈ {1024, 4096, 8192}). Phase 0 leaves 26 cells failing with `OOM`; this refinement moves them to passing. No SUPPORTED axis is added — `shape_size` is not a kernel-level branch, just a resource boundary; bucketing it would hide the gap.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: this is the per-core L1 fit refinement. Phase 0 sizes `cb_input_tiles`, `cb_centered`, `cb_centered_sq`, `cb_norm`, plus the affine RM/tile pairs as `Wt × tile_size`, which overshoots the 1.5 MB budget around `Wt=32` (W=1024) with gamma+beta and around `Wt=128` (W=4096) without. The skill's matmul-K-blocking pattern is the closest template (split the W-direction into K blocks; hold mean across blocks; recompute centered in pass 2). LayerNorm differs from matmul in that it needs two reductions (mean, then var), so the chunked pattern is: pass 1 streams the input through to compute mean; pass 2 re-streams the input, subtracting mean and squaring, to compute variance; pass 3 re-streams the input one more time with both mean and inv_std in hand to write the normalized output. Three full re-streams of the input is the cost of bounding L1 — softmax made the same trade in its Refinement 1.

**Done when**: every Phase 0 cell currently in the `OOM` category passes. The list (26 cells, all `OOM`):
- `(1,1,32,4096)` × {gamma_beta, gamma_only, no_affine}
- `(1,1,32,8192)` × {gamma_beta, gamma_only, no_affine}
- `(1,1,128,4096)` × {gamma_beta, gamma_only, no_affine}
- `(2,1,64,4096)` × {gamma_beta, gamma_only, no_affine}
- `(2,512,1024)` × {gamma_beta}
- `(1,32,4096)` × {gamma_beta, gamma_only, no_affine}
- `(1,32,8192)` × {gamma_beta, gamma_only, no_affine}
- `(1024,1024)` × {gamma_beta}
- `(32,4096)` × {gamma_beta, gamma_only, no_affine}
- `(128,8192)` × {gamma_beta, gamma_only, no_affine}

### [ ] Refinement 4 — Non-tile-aligned shapes (ROW_MAJOR)

**Goal**: add `"w_non_aligned"` and `"h_non_aligned"` to `SUPPORTED["alignment"]`. Three coordinated changes are needed:

1. **Reader**: handle a partial last-tile-row in H (read fewer than 32 sticks for the last `i`) and a partial last-tile-column in W (read fewer than `W` bytes per stick and zero-pad the L1 region so the unused lanes don't perturb the reduce).
2. **Compute / reduce**: use the partial-scaler helper variant
   (`prepare_reduce_scaler_with_partial<…>` in
   `reduce_helpers_dataflow.hpp`) — pair with
   `ReducePartialScaler::last_tile_at(1)` on the compute side — so the
   final reduce-W tile contributes only the valid lanes and the mean
   reflects the true logical W.
3. **Writer**: write a partial last stick (fewer than `W` bytes) and a partial last tile-row of sticks, mirroring the reader.

**Implementation skill**: (none in inventory — verifier-authored)

**Verifier notes**: this is *only* the ROW_MAJOR path of non-tile-aligned support. The TILE path is automatic once Refinement 2 lands (ttnn handles tile padding via the logical-vs-padded shape distinction; the kernel never sees the unpadded extent). Order this **after** Refinement 2 so the implementer doesn't have to maintain two parallel non-aligned implementations. Once both have landed, alignment becomes a fully orthogonal axis. The partial-scaler split is the same pattern as `toy_variance`'s non-aligned path — borrow that template.

**Done when**: every TARGET cell with `layout=ROW_MAJOR_LAYOUT` and `alignment ∈ {"w_non_aligned", "h_non_aligned"}` passes. (TILE-layout non-aligned cells should already be passing via Refinement 2.)
