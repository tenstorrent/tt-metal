# Operation Requirements: backward_softmax

## Definition

- **Formula**: `grad_input = output * (grad_output - sum(output * grad_output, dim))`
- **Inputs**:

  | Name | Role | Shape pattern | Description |
  |------|------|---------------|-------------|
  | `grad_output` | upstream gradient (`dy`) | `(N, C, H, W)` | float32, TILE_LAYOUT, `H % 32 == 0`, `W % 32 == 0` |
  | `output` | forward softmax output (`y`) | `(N, C, H, W)` | float32, TILE_LAYOUT, identical shape & dtype as `grad_output` |

- **Output**: `grad_input`, shape `(N, C, H, W)`, dtype `float32`, layout `TILE_LAYOUT`
- **Parameters**:

  | Name | Type | Default | Range | Description |
  |------|------|---------|-------|-------------|
  | `dim` | `int` | `-1` | `{-1, -2}` | reduction dim (W or H) |
  | `memory_config` | `ttnn.MemoryConfig` | DRAM interleaved | DRAM/L1 interleaved | output memory config |

- **PyTorch Reference**:

  ```python
  def backward_softmax_reference(grad_output, output, dim):
      s = (output * grad_output).sum(dim=dim, keepdim=True)
      return output * (grad_output - s)
  ```

- **Import Path**: `from ttnn.operations.backward_softmax import backward_softmax`
- **Function Signature**:

  ```python
  def backward_softmax(
      grad_output: ttnn.Tensor,
      output: ttnn.Tensor,
      *,
      dim: int = -1,
      memory_config: ttnn.MemoryConfig = None,
  ) -> ttnn.Tensor
  ```

- **Validation**: dtype mismatch, wrong layout, wrong rank, shape mismatch, non-tile-aligned H/W, invalid `dim` → `ValueError`.

## Phases

> **Non-regression rule**: Every phase must pass all tests from prior phases.
> **Accuracy**: PCC ≥ 0.999 to pass initially. Refinement agents may tighten tolerances and note achieved values in changelog.
> **Checkbox protocol**: Agents mark `[x]` when a phase is complete and all tests pass.

### [x] Phase 0 — Core Implementation

- **Cores**: single-core (Tensix (0,0))
- **Dtype**: float32 only
- **Layout**: TILE only
- **Memory**: DRAM or L1 interleaved (output config caller-selectable)
- **Compute config**: hard-coded `HiFi4 + fp32_dest_acc_en=True` (no caller surface)
- **Params**: `dim ∈ {-1, -2}`, `memory_config`
- **Test shapes**: `(1,1,32,32)`, `(1,1,32,256)`, `(1,1,256,32)`, `(1,1,64,128)`, `(1,1,128,64)`, `(2,4,64,128)`

### [ ] Refinement 1 — Multi-core

- Enable `split_work_to_cores(grid_size, total_lanes)` so each core processes a subset of lanes (one row of tiles for `dim=-1`, one column for `dim=-2`).
- Each core's reader uses `start_lane` (RT arg), `num_lanes` (RT arg, replacing the compile-time `total_lanes`), and walks its assigned slice of `core_group_1` and `core_group_2`.
- No inter-core communication is needed — lanes are embarrassingly parallel.
- **Note from verifier**: the program descriptor currently sets `core_grid = CoreRange(CoreCoord(0,0))` and `num_lanes` is wired as a compile-time arg. To convert: (1) replace `core_grid` with the union `all_cores` returned by `split_work_to_cores`; (2) move `num_lanes` from CT to RT and per-core; (3) split RT args by `core_group_1` / `core_group_2`. CB descriptors already use `core_ranges = core_grid` so they will fan out automatically once `core_grid` is the union.

### [ ] Refinement 2 — Compute kernel config

- Expose `compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None` on `backward_softmax(...)`.
- Default to `HiFi4 + fp32_dest_acc_en=True` (current behavior). Forward to `ComputeConfigDescriptor` in `create_program_descriptor`.
- Allow overriding `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`. The `unpack_to_dest_mode` field should also be settable per CB.
- **Note from verifier**: HiFi4 + fp32_dest_acc on Wormhole B0 is reportedly affected by hardware bug #38306. The numerical_stability.md flags this. Empirically on this branch HiFi4 outperformed HiFi3 by ~1.4× on absolute error; the bug's impact in this op is mild but exposing the config gives callers an escape hatch. **`UnpackToDestMode::UnpackToDestFp32` cannot simply be applied to all fp32 CBs** — when set on `cb_prod` it produced `inf` outputs in a verifier probe. The matmul-based REDUCE_ROW SUM path is incompatible with that mode. Apply with care, possibly only to non-matmul-input CBs.

### [ ] Refinement 3 — Tighten precision baseline / address `(dy − s)` cancellation

- Today's PCC is 0.9999 but absolute error reaches ~0.3 at positions where `dy_i ≈ s` because the matmul-based REDUCE_ROW SUM accumulates with worse-than-fp32 effective precision (likely due to the SrcA TF32 path on Wormhole, see numerical_stability.md).
- Goals: (a) make the spec test pass (`atol=0.01, rtol=0.05`) for all currently-defined shapes; (b) document the achievable precision floor.
- **Approaches to evaluate**:
  - Re-formulate to avoid the matmul reduce path for `dim=-1` (e.g. transpose to `dim=-2` and back, since REDUCE_COL uses non-matmul `reduce_tile`).
  - Use `transform_in_place` on `cb_sum` to add a small post-pass correction (Kahan-style compensation).
  - Lower-precision pass-2 sub on a re-summed-in-fp32-CPU-by-default path.
- **Note from verifier**: this is the single biggest gap between what the spec test asserts and what the operation delivers. Without it the operation is "correct in PCC, off by hardware-precision-floor in atol".

### [ ] Refinement 4 — Float32 alternative dtypes (BFLOAT16, BFLOAT8_B)

- Add support for bfloat16 and bfloat8_b inputs/outputs.
- Both inputs must match dtype. The output dtype matches the input.
- Update CB formats and tile sizes accordingly.
- The matmul reduce already accepts mixed scaler (bf16) + fp32 inputs; switching to bf16 inputs will need a bf16 scaler too.

### [ ] Refinement 5 — Non-tile-aligned shapes (`H` or `W` % 32 ≠ 0)

- Today's validator rejects non-aligned. The reader assumes tile-aligned per-lane tile counts; the reduce assumes the scaler lands on the first column / row of a single tile.
- Add a partial-scaler path: `dataflow_kernel_lib::prepare_partial_reduce_scalers` plus `ReducePartialScaler::last_tile_at(1)` in the compute kernel, like `toy_variance` does.
- Ensure the partial-scaler tile zeros the padded positions so the reduce sees `0 × y_pad` for those lanes.
- **Note from verifier**: this is straightforward — toy_variance is a textbook reference, all the helper plumbing exists. Mostly bookkeeping.

### [ ] Refinement 6 — ROW_MAJOR input support

- Today's validator rejects ROW_MAJOR layout. Add an in-kernel tilize / fused tilize-then-compute path (or rely on host-side `to_layout(TILE)` being implicit when the API accepts ROW_MAJOR).
- Probably easier to call host-side `to_layout` inside the entry point and document the cost.

### [ ] Refinement 7 — Rank flexibility

- Today's validator rejects rank ≠ 4. PyTorch supports any rank (with the reduction dim valid for that rank).
- Internally reshape to `(N, C, H, W)` (e.g. by squeezing to 4D via leading-1 dims) before launch. This is a pure entry-point change — no kernel changes — as long as `dim ∈ {-1, -2}` corresponds to W/H of the reshaped 4D tensor.

### [ ] Refinement 8 — Additional `dim` values (`dim=-3`, `dim=-4`, positive dims)

- Today's validator rejects all dims except `{-1, -2}`. The kernel only knows two reader formulas (W reduce, H reduce).
- For other dims, the natural implementation is to permute the tensor so the reduction axis lands on the last axis, run the existing kernel, then permute back. Or write new reader formulas.
- Probably easiest as a permute + recurse-on-self.

### [ ] Refinement 9 — Memory pressure (sharded inputs/outputs, large shapes)

- Today the operation always launches with interleaved DRAM/L1 inputs and outputs.
- For sharded inputs, the reader's `TensorAccessor` pattern works without changes if the shard layout matches lane decomposition; otherwise the reader needs core-local lane→tile_id mapping.
- Sharded outputs would need writer changes.
- This is the last refinement because every other refinement is composed under the assumption of interleaved memory.
