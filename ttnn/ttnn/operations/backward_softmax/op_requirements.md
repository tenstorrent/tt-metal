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

### [x] Refinement 1 — Multi-core distribution (balanced load across the full grid)

- **Goal**: distribute the independent lanes (one tile-row per lane for `dim=-1`, one tile-column for `dim=-2`) across **all cores in the chip's compute-with-storage grid**, with load balanced so that no two cores differ by more than 1 lane. This is the operative contract — the implementation must satisfy it, not approximate it.
- **Properties**: lanes are embarrassingly parallel — no inter-core communication, no work-stealing, no sub-lane splitting. Each core runs its assigned lanes end-to-end.
- **Grid**: use `device.compute_with_storage_grid_size()` for the full available grid; do not hard-code a subset.
- **Tests**: add shape(s) whose lane count is not a clean multiple of the grid (forcing the remainder distribution to kick in). Existing Phase-0 tests must continue to pass.
- **Verifier notes (current Phase-0 state)**:
  - Program descriptor pins `core_grid = CoreRange(CoreCoord(0,0))` and passes `total_lanes` as a compile-time arg.
  - To convert: (1) replace `core_grid` with the union of all cores in the grid; (2) move `num_lanes` and `start_lane` from CT to per-core RT args; (3) the CB descriptors already key off `core_ranges = core_grid` so they will fan out automatically once `core_grid` is the union.

### [x] Refinement 2 — Choose input-buffering strategy by shape and L1 budget

- **Goal**: minimize DRAM bandwidth by holding `grad_output` and `output` tiles in L1 across both passes when possible. The strategy must be **chosen at program-descriptor time** as a function of shape (dtype × reduce_dim_tiles) and the per-core L1 budget — no runtime probing.
- **Why it matters**: the current Phase-0 reader fetches each input tile twice per output tile (once for the multiply-then-reduce pass, once for the subtract-then-multiply pass). DRAM bandwidth is the gating resource for memory-bound ops like this one — eliminating the second read is the single biggest perf lever after multi-core.
- **Three known-good strategies, in order of preference when L1 allows**:
  1. **Whole row, double-buffered.** Cache CBs sized to `2 × reduce_dim_tiles` pages each. Each input tile is read from DRAM exactly once per output tile, AND the reader can stage lane N+1 while compute processes lane N — overlapping DRAM read latency with compute.
  2. **Whole row, single-buffered.** Cache CBs sized to `reduce_dim_tiles` pages each. Each input tile is read from DRAM exactly once per output tile. Reader and compute alternate per lane (no cross-lane overlap), but DRAM traffic is already halved vs Phase 0.
  3. **Per-tile streaming (fallback).** Cache CBs sized to a small constant (the Phase-0 behavior). Each tile is read twice — once per pass — but the per-core working set stays tiny. This is the fallback for shapes whose reduce dimension is too large to cache in L1 under strategies 1 or 2.
- **Selection contract**: pick the deepest strategy whose per-core working set fits within the chosen L1 budget. The choice must be deterministic given shape, dtype, and grid. **Do not fail** on shapes too large for strategies 1 or 2 — degrade to strategy 3.
- **L1 budget (rough, per Tensix core on Wormhole B0)**: ~1 MB usable after firmware/dispatch overhead. The agent must pick a conservative number that leaves headroom for output / scaler / intermediate CBs, kernel stacks, and any helper-allocated scratch; an off-by-2× overestimate hangs the device at CB allocation. Document the chosen budget in `capabilities.md`.
- **Numerical correctness**: must continue to satisfy the Phase-0 contract (PCC and atol). Precision is the subject of Refinement 4, not this one.
- **Tests**:
  - Existing acceptance + extended + precision-baseline must pass unchanged.
  - Add a shape that triggers strategy 1 (small `reduce_dim_tiles`).
  - Add a shape that triggers strategy 2 (medium `reduce_dim_tiles`, fits single-buffered but not double-buffered).
  - If no in-scope shape triggers strategy 3, that's fine — note in `capabilities.md` that strategy 3 is reachable only at very large reduce dimensions and is not exercised by the test set.
- **Verifier hint**: the standard TTNN pattern for keeping a CB live across multiple compute passes is to `cb_wait_front(cb, N)` once at the top and only `cb_pop_front(cb, N)` after the last pass that reads it. Compute APIs that take a tile-index argument read at arbitrary offsets within the waited region.

### [x] Refinement 3 — Alternative input dtypes (BFLOAT16, BFLOAT8_B)

- Add support for `bfloat16` and `bfloat8_b` inputs/outputs in addition to the current `float32`.
- Both `grad_output` and `output` inputs must have the same dtype. The returned `grad_input` dtype matches the input dtype.
- Update CB formats and tile sizes accordingly. The scaler CB is bf16 regardless of input dtype (matmul reduce path accepts mixed `bf16` scaler + any-dtype data).
- **Dtype-aware default compute config**: pick sensible per-dtype defaults rather than reusing the fp32 settings for bf16/bfp8. The current fp32 defaults (HiFi4 + fp32_dest_acc_en=True) are precision-first; for bf16/bfp8 storage the input is already quantized to ~3 decimal digits, so HiFi4 overhead buys little and a lower-fidelity / approx-mode default is the better trade-off. Pick defaults that match the operation's chosen design point (see how `glu_fused` handled this in its R2). Refinement 4 below will let callers override the chosen defaults explicitly.
- **Tests**: parametrize the acceptance + extended test set over `{float32, bfloat16, bfloat8_b}`. Expect PCC ≥ 0.999 across dtypes; absolute-error tolerances may need loosening for bf16/bfp8 (document the chosen tolerances per dtype in `capabilities.md`).

### [x] Refinement 4 — Compute kernel config exposed to caller

- Expose `compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None` on `backward_softmax(...)`.
- Default to the dtype-aware config chosen in Refinement 3 (no behavior change when `None` is passed). Forward to `ComputeConfigDescriptor` in `create_program_descriptor`.
- Allow overriding **all** of the following `ComputeConfigDescriptor` fields:
  - `math_fidelity` (HiFi4 / HiFi3 / HiFi2 / LoFi)
  - `fp32_dest_acc_en` (bool)
  - `math_approx_mode` (bool — affects SFPU op accuracy)
  - `dst_full_sync_en` (bool)
  - `unpack_to_dest_mode` per CB (vector of `UnpackToDestMode`)
- **Tests**: at least one parametrized config case per dtype (e.g., `HiFi2 + fp32_dest_acc + math_approx=True` on bf16, exercising the perf-first regime callers would pick for throughput-bound workloads).
- **Note from verifier**: HiFi4 + fp32_dest_acc on Wormhole B0 is reportedly affected by hardware bug #38306 — exposing the config gives callers an escape hatch. **`UnpackToDestMode::UnpackToDestFp32` cannot simply be applied to all fp32 CBs** — when set on `cb_prod` it produced `inf` outputs in a verifier probe. The matmul-based REDUCE_ROW SUM path is incompatible with that mode. Apply with care, possibly only to non-matmul-input CBs.

### [ ] Refinement 5 — Tighten precision baseline / address `(dy − s)` cancellation

- Today's PCC is 0.9999 but absolute error reaches ~0.3 at positions where `dy_i ≈ s` because the matmul-based REDUCE_ROW SUM accumulates with worse-than-fp32 effective precision (likely due to the SrcA TF32 path on Wormhole, see numerical_stability.md).
- Goals: (a) make the spec test pass (`atol=0.01, rtol=0.05`) for all currently-defined shapes; (b) document the achievable precision floor.
- **Approaches to evaluate**:
  - Re-formulate to avoid the matmul reduce path for `dim=-1` (e.g. transpose to `dim=-2` and back, since REDUCE_COL uses non-matmul `reduce_tile`).
  - Use `transform_in_place` on `cb_sum` to add a small post-pass correction (Kahan-style compensation).
  - Lower-precision pass-2 sub on a re-summed-in-fp32-CPU-by-default path.
- **Note from verifier**: this is the single biggest gap between what the spec test asserts and what the operation delivers. Without it the operation is "correct in PCC, off by hardware-precision-floor in atol".

### [ ] Refinement 6 — Non-tile-aligned shapes (`H` or `W` % 32 ≠ 0)

- Today's validator rejects non-aligned. The reader assumes tile-aligned per-lane tile counts; the reduce assumes the scaler lands on the first column / row of a single tile.
- Add a partial-scaler path: `dataflow_kernel_lib::prepare_partial_reduce_scalers` plus `ReducePartialScaler::last_tile_at(1)` in the compute kernel, like `toy_variance` does.
- Ensure the partial-scaler tile zeros the padded positions so the reduce sees `0 × y_pad` for those lanes.
- **Note from verifier**: this is straightforward — toy_variance is a textbook reference, all the helper plumbing exists. Mostly bookkeeping.

### [ ] Refinement 7 — ROW_MAJOR input support

- Today's validator rejects ROW_MAJOR layout. Add an in-kernel tilize / fused tilize-then-compute path (or rely on host-side `to_layout(TILE)` being implicit when the API accepts ROW_MAJOR).
- Probably easier to call host-side `to_layout` inside the entry point and document the cost.

### [ ] Refinement 8 — Rank flexibility

- Today's validator rejects rank ≠ 4. PyTorch supports any rank (with the reduction dim valid for that rank).
- Internally reshape to `(N, C, H, W)` (e.g. by squeezing to 4D via leading-1 dims) before launch. This is a pure entry-point change — no kernel changes — as long as `dim ∈ {-1, -2}` corresponds to W/H of the reshaped 4D tensor.

### [ ] Refinement 9 — Additional `dim` values (`dim=-3`, `dim=-4`, positive dims)

- Today's validator rejects all dims except `{-1, -2}`. The kernel only knows two reader formulas (W reduce, H reduce).
- For other dims, the natural implementation is to permute the tensor so the reduction axis lands on the last axis, run the existing kernel, then permute back. Or write new reader formulas.
- Probably easiest as a permute + recurse-on-self.

### [ ] Refinement 10 — Memory pressure (sharded inputs/outputs, large shapes)

- Today the operation always launches with interleaved DRAM/L1 inputs and outputs.
- For sharded inputs, the reader's `TensorAccessor` pattern works without changes if the shard layout matches lane decomposition; otherwise the reader needs core-local lane→tile_id mapping.
- Sharded outputs would need writer changes.
- This is the last refinement because every other refinement is composed under the assumption of interleaved memory.
