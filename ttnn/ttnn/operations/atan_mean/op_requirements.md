# Operation Requirements: atan_mean

## Definition

- **Formula**: `output[n, c, h] = (1 / W) · Σ_{w=0..W-1} atan(input[n, c, h, w])` — equivalently `torch.atan(input).mean(dim=-1)`.
- **Inputs**:

  | Name | Role | Shape pattern | Description |
  |------|------|---------------|-------------|
  | `input_tensor` | Sole input | `(N, C, H, W)` rank 4 | fp32 TILE_LAYOUT on device, with `H % 32 == 0` and `W % 32 == 0`. |

- **Output**: Rank-3 `(N, C, H)` float32, TILE_LAYOUT, DRAM interleaved (returned post-squeeze; allocated internally as rank-4 `(N, C, H, 1)`).
- **Parameters**: None beyond `input_tensor`.

  | Name | Type | Default | Range | Description |
  |------|------|---------|-------|-------------|
  | — | — | — | — | No additional parameters in Phase 0. |

- **PyTorch Reference**:

  ```python
  def torch_atan_mean(x: torch.Tensor) -> torch.Tensor:
      return torch.atan(x).mean(dim=-1)
  ```

- **Import Path**: `from ttnn.operations.atan_mean import atan_mean`
- **Function Signature**: `atan_mean(input_tensor: ttnn.Tensor) -> ttnn.Tensor`
- **Validation**: rejects non-device tensors, non-fp32 dtype, non-TILE layout, rank ≠ 4, H not divisible by 32, W not divisible by 32.

## Phases

> **Non-regression rule**: Every phase must pass all tests from prior phases.
> **Accuracy**: PCC > 0.99 to pass initially. Agents tighten tolerances and note achieved values in changelog.
> **Checkbox protocol**: Agents mark `[x]` when a phase is complete and all tests pass.

### [x] Phase 0 — Core Implementation
- **Cores**: Multi-core (`ttnn.split_work_to_cores` across `compute_with_storage_grid_size()`).
- **Dtype**: float32 only.
- **Layout**: TILE_LAYOUT only.
- **Memory**: DRAM interleaved (input may be L1 interleaved or DRAM; output always DRAM).
- **Compute config**: Hard-coded `HiFi4` + `fp32_dest_acc_en=True`; not user-configurable.
- **Params**: `input_tensor` only.
- **Test shapes**: `(1,1,32,32)`, `(1,1,2048,64)`, `(1,1,1024,64)`, `(1,1,2048,32)`, `(1,1,1024,32)`, `(1,256,64,64)`, `(256,1,64,64)`, `(1,128,128,128)`, `(128,1,128,128)`.
- **Achieved accuracy**: PCC ≥ 0.99999995, max-abs ≈ 1.3e-4 (well inside Phase 0 acceptance: PCC ≥ 0.9995, max-abs ≤ 1e-2).

### [ ] Refinement 1 — Expose `compute_kernel_config`
- **Why**: Phase 0 hard-codes `HiFi4 + fp32_dest_acc_en=True`. Callers cannot trade accuracy for throughput. The numerical-stability analysis flags these as the primary precision levers.
- **What changes**:
  - Add `compute_kernel_config: ttnn.WormholeComputeKernelConfig | ttnn.GrayskullComputeKernelConfig | None = None` to the `atan_mean(...)` signature.
  - In `create_program_descriptor`, derive `math_fidelity`, `fp32_dest_acc_en`, and `math_approx_mode` from the supplied config (with current Phase-0 values as defaults when `None` is passed).
- **Tests**: Add at least one parametrized config case (e.g., `HiFi3 + fp32_dest_acc_en=True`, `HiFi2 + fp16b_dest_acc`) verifying PCC ≥ 0.999 on the existing shape set.
- **Notes from verifier**:
  - When `fp32_dest_acc_en=False`, the effective DEST capacity doubles to 8 tiles. `sfpu_atan` auto-batching honors this via `DEST_AUTO_LIMIT` — no kernel changes needed.
  - HiFi3 with fp32 dest is the documented workaround for Wormhole-B0 HW issue #38306, which the Phase-0 HiFi4+fp32 combo nominally hits but the acceptance tests do not trip. Document this in changelog.
  - The matmul-mode REDUCE_ROW path through `reduce_helpers_compute.inl` already respects whatever fidelity the kernel was compiled with — no path changes.

### [ ] Refinement 2 — Support bfloat16 (and bfloat8_b) input dtype
- **Why**: PyTorch users routinely run atan/mean on bf16 in mixed-precision pipelines. The acceptance test currently rejects bf16.
- **What changes**:
  - Relax the dtype validator in `_validate_input` to accept `bfloat16` (and optionally `bfloat8_b`).
  - Adjust `cb_input_tiles` and `cb_atan_tiles` page sizes via `input_tensor.buffer_page_size()` — they already key off `input_tensor.dtype`, so the program descriptor needs no further changes.
  - Allocate the output tensor with the same dtype as the input (currently `input_tensor.dtype`, already correct).
- **Tests**: Parametrize the acceptance-style shapes over `{float32, bfloat16}`. With bf16 accumulation depth W, expect PCC ≥ 0.999 (not 0.9999).
- **Notes from verifier**:
  - The bf16 scaler tile format is fixed regardless of input dtype (matmul col-0 convention) — no scaler-format branching needed.
  - With bfloat16 input + `fp32_dest_acc_en=True`, the matmul ingress unpacks bf16 → fp32 on SrcA, so accumulation precision is unchanged from fp32 input. Numerical degradation comes solely from the input-tensor dtype, not the reduce path.

### [ ] Refinement 3 — Support non-tile-aligned H and W
- **Why**: Phase 0 hard-rejects `H % 32 != 0` or `W % 32 != 0`. The kernel-lib already exposes the partial-scaler infrastructure (`calculate_and_prepare_partial_reduce_scalers`, `ReducePartialScaler::last_tile_at`); it just needs wiring.
- **What changes**:
  - Drop the H/W tile-alignment validation in the entry point.
  - In the program descriptor, compute `partial_w = W % 32`. When non-zero:
    - Switch the reader's scaler emission to `calculate_and_prepare_partial_reduce_scalers<cb_scaler, AVG, REDUCE_ROW, partial_w>(scaler_value)`.
    - Switch the compute's `reduce<>` call to pass `ReducePartialScaler::last_tile_at(1)`.
  - Reader's W loop must clamp to the logical W on the last W-tile (or rely on the partial scaler zeroing the padded columns).
  - Read `W` from `input_tensor.logical_shape()[-1]` rather than the padded shape so the scaler's `1/W` matches the *valid* element count.
  - For non-aligned H, the row-tile work distribution already handles padded rows correctly because the writer's output_tile_id = row-tile index — the last H-tile's padded rows simply land in unused output positions that the squeezed-rank-3 view never observes.
- **Tests**: Parametrize over shapes like `(1,1,32,30)`, `(1,1,32,48)`, `(1,1,30,32)`, `(1,1,30,30)`, `(1,1,48,96)`.
- **Notes from verifier**:
  - `numerical_stability.md` §"Numerical Guards" specifically calls out this wiring path with the exact helper signatures to use.
  - Scaler factor must come from the **logical** W (not padded W) — easy to get wrong.

### [ ] Refinement 4 — Support ROW_MAJOR_LAYOUT input
- **Why**: Phase 0 hard-rejects ROW_MAJOR input. Callers must `.to_layout(TILE_LAYOUT)` host-side, which dispatches a separate program.
- **What changes**:
  - Add an in-reader tilize stage using `tilize_helpers_dataflow.hpp` (RM → TILE in the reader) before pushing into `cb_input_tiles`.
  - Or, dispatch a different reader kernel when the input is RM and fold the tilize into the data pipeline.
  - Validation: accept both TILE_LAYOUT and ROW_MAJOR_LAYOUT in `_validate_input`.
- **Tests**: One ROW_MAJOR shape per existing acceptance shape, verifying numerical match with TILE_LAYOUT equivalent.

### [ ] Refinement 5 — Support arbitrary rank (rank != 4)
- **Why**: Phase 0 fixes rank to 4. PyTorch supports any rank.
- **What changes**:
  - In `_validate_input`, accept any rank ≥ 1.
  - Reshape to rank-4 `(N, C, H, W)` internally by collapsing leading dims into N — e.g., for input shape `(D0, D1, ..., Dk, W)`, set `N = D0`, `C = D1`, `H = Π_{i≥2, i≤k-1} Di × Dk`. Or, follow the simpler pattern: pad to rank-4 by inserting `1`s for missing leading dims, fold trailing dims into H.
  - Output rank is `input.rank - 1` (drop the W dim).
- **Tests**: Rank 1 (`(64,)`), rank 2 (`(32, 64)`), rank 3 (`(2, 32, 64)`), rank 5 (`(1, 2, 1, 32, 64)`).

### [ ] Refinement 6 — Memory pressure: support large W
- **Why**: `cb_atan_tiles` is sized to `Wt = W/32` fp32 pages. For W = 8192 → Wt = 256 → 1 MB just for this CB on every core. With other CBs and kernel stacks, the 1.5 MB L1 budget runs out somewhere around W=4096.
- **What changes**:
  - Replace the "full row in `cb_atan_tiles` then reduce" pipeline with a streaming reduce: chunk the row into K blocks of size `B` tiles each, where `B` is L1-friendly (e.g., 4 tiles), and use `streaming_reduce_helpers::accumulate_reduce<AVG, REDUCE_ROW>(cb_atan_tiles, cb_scaler, cb_acc, row(B), K)`.
  - `cb_atan_tiles` then holds `B` pages instead of `Wt`.
  - The accumulating-reduce helper already handles partial-block scaler routing for the last block when `W % (B*32) != 0`.
- **Tests**: Parametrize over W ∈ {1024, 2048, 4096, 8192} once L1 ceiling is lifted.
- **Notes from verifier**:
  - This is the **last refinement** because memory pressure issues are typically the gating constraint after all functional features are in place.
  - The `streaming_reduce_helpers::accumulate_reduce` helper is the canonical wrapper for this pattern. Reference `toy_variance` for the precedent.
