# Operation Requirements: rms_norm

## Definition

- **Formula**: `output[..., i, j] = input[..., i, j] / sqrt(mean(input[..., i, :]^2) + epsilon) * gamma[j]`
  (gamma defaults to 1.0 when `gamma is None`).
- **PyTorch Reference** (standalone, in fp32):

  ```python
  def torch_rms_norm(x, gamma=None, eps=1e-6):
      x_fp = x.float()
      rms = torch.sqrt(x_fp.pow(2).mean(dim=-1, keepdim=True) + eps)
      out = x_fp / rms
      if gamma is not None:
          out = out * gamma.float().view(*([1] * (x.dim() - 1)), -1)
      return out.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  def rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      epsilon: float = 1e-6,
  ) -> ttnn.Tensor: ...
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update `SUPPORTED`. The implementer fixes by updating `SUPPORTED`.
> **Checkbox protocol**: `[x]` when the refinement is complete and all tests pass; `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial); `[ ]` only when nothing usable was produced.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: `[bfloat16, float32]`
- **SUPPORTED layout**: `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]`
- **SUPPORTED alignment**: `[tile_aligned, w_non_aligned, h_non_aligned]`
  (TILE_LAYOUT × non-aligned excluded via EXCLUSIONS — TILE requires
  H, W divisible by 32; the partial-W scaler and writer-side partial-H
  stick skip are RM-only paths.)
- **SUPPORTED rank**: `[2, 3, 4]`
- **SUPPORTED shape_size**: `[small]` (Wt ≤ 32 ⇔ W ≤ 1024)
- **SUPPORTED gamma_mode**: `[gamma, no_gamma]`
- **SUPPORTED gamma_dtype**: `[bfloat16, float32]` (independent of input dtype, except as restricted by EXCLUSIONS for the TILE-input path — see Refinement 3)
- **SUPPORTED gamma_layout**: `[ROW_MAJOR_LAYOUT, TILE_LAYOUT]`
  (TILE_LAYOUT is in SUPPORTED only to admit the canonical no_gamma
  cell from `feature_spec.INVALID`; when gamma is actually supplied,
  TILE-layout gamma is EXCLUSIONS-gated.)
- **Cores**: single-core (`CoreRangeSet([CoreRange((0,0),(0,0))])`).
- **Compute config**: `ComputeConfigDescriptor(fp32_dest_acc_en=(input_dtype==fp32))`.
  `math_fidelity` defaults to HiFi4. Not user-configurable.
- **Golden baseline**: 210 / 2535 cells passing inside SUPPORTED;
  840 xfail (outside SUPPORTED or in EXCLUSIONS), 1470 INVALID-skipped.
  All loud verifier categories (xpass_drift, supported_fail,
  xfail_wrong_mode, supported_marked_xfail, invalid_unexpected) are 0.
- **Precision baseline**: bf16 PCC ≥ 0.995 (max_abs ≈ 0.03–0.09);
  fp32 PCC ≥ 0.999 (max_abs ≈ 0.01–0.03). Both layouts tested.

### [ ] Refinement 1 — Wide-W support via W-blocking (`shape_size=large`)

**Goal**: add `large` to `SUPPORTED["shape_size"]` so the op handles
`W > 1024` (`Wt > 32`) without blowing the per-core L1 budget.

Bundles in the multi-core distribution: when W is wide each row chunk
benefits substantially from parallelism, and the multi-core path is
embarrassingly parallel (one row chunk per core via
`split_work_to_cores`). Doing both at once avoids a v0.5 step where
multi-core lands but the L1 cap still gates the very shapes that
multi-core would benefit most.

**Mechanics**:

- Switch the reduce path from `BulkWaitBulkPop` (which holds all
  `Wt` tiles in `cb_x_sq` at once) to a block-streaming reduce via
  `accumulate_reduce_block` from
  `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:53-61`.
  Pick a `BLOCK_SIZE` of 8–16 tiles (well under any plausible
  `DEST_AUTO_LIMIT`); the CB sizing for `cb_x_sq` drops to
  `BLOCK_SIZE * tile_size` instead of `Wt * tile_size`.
- Stage D (`x · rsqrt`) and Stage E (`x_norm · gamma`) already stream
  one tile at a time per chain iteration, so `cb_input_tiles`,
  `cb_x_norm`, and `cb_output_tiles` can shrink to `BLOCK_SIZE`-sized
  buffers too. The only invariant to preserve is "cb_input_tiles
  holds Wt across pass 1 and pass 2" — break that by giving the reader
  a two-pass push schedule (or by tilizing the RM input twice).
- Single-core → multi-core: `ttnn.split_work_to_cores(grid, num_chunks)`
  yields per-core start_chunk + num_chunks. Reader/writer/compute each
  receive those as runtime args (compute only needs num_chunks; reader
  and writer use start_chunk for `TensorAccessor` offsets — already
  scaffolded as `input_start_unit` / `output_start_unit` in the
  descriptor).
- `cb_input_raw_rm` / `cb_output_rm` similarly shrink to `BLOCK_SIZE`-
  sized buffers if the tilize/untilize helpers are called per-block.

**Verifier notes**:

- This is the **algorithm-fundamental refinement** for this op —
  rewrites the data pipeline.
- `data_transfer.md` shows the L1 totals exceed 1.5 MB starting at
  W=4096 fp32 (~3.6 MB) and W=4096 bf16 (~1.7 MB), so blocking
  must apply on both dtypes.
- Acceptance test (`test_rms_norm.py`) only goes up to W=1024 and
  doesn't gate any multi-core behavior — add new acceptance cases
  for W=2048, W=4096, W=8192 (matched-dtype gamma) once this lands.
- After this lands, `shape_size` axis can be deleted from SUPPORTED
  if the kernel is genuinely shape-agnostic; otherwise simply
  promote `large` into the list.

### [ ] Refinement 2 — `compute_kernel_config` parameter

**Goal**: extend the public signature to accept `compute_kernel_config:
Optional[ttnn.ComputeConfigDescriptor] = None` and thread it through to
the program descriptor. Adds nothing to `SUPPORTED` directly — but it
**unlocks a future axis** that the golden suite will introduce when
mixed math-fidelity coverage lands. For now this refinement is queued
because it's the cleanest unlocker for the next family of refinements
(precision tradeoffs, Wormhole HiFi4+fp32-dest workaround).

If/when `feature_spec.TARGET` gains a `math_fidelity` or
`fp32_dest_acc_en` axis, this refinement's deliverable is "add
`HiFi2 / HiFi3 / HiFi4` to `SUPPORTED["math_fidelity"]` and
`{True, False}` to `SUPPORTED["fp32_dest_acc_en"]`."

**Verifier notes** (from `numerical_stability.md`):

- Wormhole B0 fp32 path currently hits HiFi4 + `fp32_dest_acc_en=True`
  — bug #38306. Either accept caller-pinned `HiFi3` here, or cap the
  default to HiFi3 inside the descriptor when fp32_dest is on.
- HiFi2 on normalization workloads is usually safe and is a 4×
  FPU-cost reduction vs HiFi4. Worth recommending in the docstring.
- Goal in axis-add terms: declare `math_fidelity` / `math_approx_mode`
  / `fp32_dest_acc_en` as op-specific axes whose SUPPORTED list
  starts at `{HiFi4}` / `{False}` / derived-from-dtype respectively
  (Phase 0 behaviour) and grows.

> **Sequencing**: this is *not* the algorithm-fundamental refinement.
> Sequencing 2 before 1 keeps the perf-knob exposed to the refinement
> implementer who lands W-blocking — they may want to pin HiFi3 to
> work around the WH B0 bug while iterating, rather than racing the
> two changes.

### [ ] Refinement 3 — Mixed-precision gamma on the TILE input path

**Goal**: remove the two EXCLUSIONS entries

```python
{"layout": TILE_LAYOUT, "gamma_mode": "gamma",
 "dtype": float32,  "gamma_dtype": bfloat16},
{"layout": TILE_LAYOUT, "gamma_mode": "gamma",
 "dtype": bfloat16, "gamma_dtype": float32},
```

so `gamma_dtype` can differ from `dtype` on the TILE-input path
(matched-dtype already works on TILE, and mixed-dtype already works on
RM input). Unlocks the LLM-realistic case of bf16 activations + fp32
weights — without restricting users to pre-cast their gamma to match.

**Verifier notes**:

- Residual after the Phase 0 Stage A reconfig fix is a uniform
  ~1.27× over-amplification of the output (probe_005). The chain's
  Stage E reconfig already emits the necessary srcA/srcB/pack format
  changes per `eltwise_chain.inl:452-460` — but `unpack_to_dest_mode`
  on `cb_x_norm` / `cb_gamma_tiled` / `cb_output_tiles` is the default,
  not `UnpackToDestMode::UnpackToDestFp32`. `numerical_stability.md`
  flags exactly this as the residual fp32 precision wall (point 4).
  First thing to try: set `UnpackToDestMode::UnpackToDestFp32` on the
  fp32-side CBs and rerun the failing 24 cells.
- If that doesn't fully close the gap, suspect:
  - Asymmetric tilize of bf16 gamma leaving the unpack format
    set for bf16 across the chunk loop (the chain's first iteration
    reconfigs, but per-chunk re-entry may not).
  - `binary_op_init_common` not being called between Phase 0 and
    Stage A on the TILE path (the RM path's Phase 1a tilize re-issues
    init, masking the issue).
- Add an acceptance-test case (matched + mismatched dtype, TILE
  input, ranks 2/3/4) to `test_rms_norm.py` once this lands. The
  current acceptance test only exercises matched gamma_dtype.
- Removing the EXCLUSIONS entries is the only edit to the op file
  required for this refinement, after the kernel fix is verified
  by the golden suite (the 24 failed cells become `supported_pass`).
