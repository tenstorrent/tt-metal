# Operation Design: layer_norm

## Overview
- **Operation Name**: layer_norm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), softmax (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[i] = sum(x[i, :]) / W
var[i] = sum((x[i, :] - mean[i])^2) / W
y[i, j] = (x[i, j] - mean[i]) / sqrt(var[i] + eps) * gamma[j] + beta[j]
```
Layer normalization over the last dimension (W) of a 2D input. Mean and variance use biased estimator (divide by W). Gamma and beta are optional 1D affine parameters.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| eps | float | No | > 0 | 1e-5 | Numerical stability constant |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| x shape | 2D (N, W) | Must be rank 2 |
| x layout | ROW_MAJOR | Input must be row-major |
| x memory | INTERLEAVED | Must be interleaved DRAM |
| x dtype | BFLOAT16 or FLOAT32 | Unsupported dtype |
| W | multiple of 32 | W must be tile-aligned |
| N | multiple of 32 | N must be tile-aligned (for tilize) |
| weight shape | (W,) or None | Must match last dim |
| bias shape | (W,) or None | Must match last dim |

### Output Tensor Specification
- **Shape**: same as x (N, W)
- **Dtype**: same as x
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| weight=None, bias=None | Skip affine transform |
| weight provided, bias=None | Apply only gamma scaling |
| eps very small | May cause numerical issues; user responsibility |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add weight/bias reading, generate reduce scaler and eps |
| Compute | softmax (W-small) | compute_core | Replace max/exp/sum with mean/sub/square/var/rsqrt/affine |
| Writer | untilize | output_stage | None |

### Work Distribution
- **Work unit**: one tile-row (Wt tiles, covering 32 rows of input)
- **Grid**: 1x1 (single core)
- **Work per core**: all Ht tile-rows (Ht = N / 32)
- **Remainder**: N/A (single core)

### Data Flow
Reader tilizes RM input into cb_in (Wt tiles per row). Compute performs mean, subtract, square, variance, rsqrt, normalize, and optional affine. Writer untilizes output from cb_out back to RM sticks and writes to DRAM. W-small variant: all Wt tiles for one row fit in L1 simultaneously.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in | Tilized input (one row) | Reader | Compute | Wt | Row |
| c_1 | cb_eps | Epsilon tile (scalar) | Reader | Compute | 1 | Program |
| c_2 | cb_scaler | Reduce scaler (1.0) | Reader | Compute | 1 | Program |
| c_3 | cb_weight | Gamma tiles (one row) | Reader | Compute | Wt | Program |
| c_4 | cb_bias | Beta tiles (one row) | Reader | Compute | Wt | Program |
| c_16 | cb_tilize_out | Tilize output / compute input | Compute (tilize) | Compute (reduce) | Wt | Row |
| c_17 | cb_out | Final output (tiled, for untilize) | Compute | Compute (untilize) | Wt | Row |
| c_18 | cb_untilize_out | Untilized output (RM) | Compute (untilize) | Writer | Wt | Row |
| c_24 | cb_mean | Row mean (REDUCE_ROW output) | Compute | Compute | 1 | Row |
| c_25 | cb_x_minus_mean | x - mean intermediates | Compute | Compute | Wt | Row |
| c_26 | cb_sq | (x - mean)^2 intermediates | Compute | Compute | Wt | Row |
| c_27 | cb_var | Variance (REDUCE_ROW output) | Compute | Compute | 1 | Row |
| c_28 | cb_inv_std | 1/sqrt(var + eps) scalar | Compute | Compute | 1 | Row |
| c_29 | cb_norm | Normalized output | Compute | Compute | Wt | Row |

Intermediate CBs (c_24-c_29) use Float32 format when `fp32_dest_acc_en` is true, otherwise match input format.

### Kernel Arguments

**Compile-time** (reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | Bytes per input row (W * elem_size) |
| Reader | 1+ | TensorAccessorArgs (src) | uint32_t[] | Input buffer accessor |

**Runtime** (reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer DRAM address |
| Reader | 1 | num_rows | uint32_t | Total rows (N) |
| Reader | 2 | Wt | uint32_t | Tiles per row (W / 32) |
| Reader | 3 | block_width_size | uint32_t | W * elem_size per tile-row block |
| Reader | 4 | has_weight | uint32_t | 1 if gamma provided |
| Reader | 5 | has_bias | uint32_t | 1 if beta provided |
| Reader | 6 | weight_addr | uint32_t | Weight buffer address (0 if none) |
| Reader | 7 | bias_addr | uint32_t | Bias buffer address (0 if none) |
| Reader | 8 | eps | uint32_t | Epsilon as bit-cast float |

**Compile-time** (compute):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Compute | 0 | Ht | uint32_t | Tile-rows to process (N / 32) |
| Compute | 1 | Wt | uint32_t | Tiles per row (W / 32) |
| Compute | 2 | has_weight | uint32_t | 1 if gamma provided |
| Compute | 3 | has_bias | uint32_t | 1 if beta provided |

**Compile-time** (writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | cb_id_out | uint32_t | Output CB (c_18) |
| Writer | 1 | output_stick_size | uint32_t | Output row size in bytes |
| Writer | 2 | tile_height | uint32_t | 32 |
| Writer | 3 | Ht | uint32_t | Tile-rows |
| Writer | 4 | num_tiles_per_block | uint32_t | Wt |
| Writer | 5 | block_width_size | uint32_t | Wt * 32 * elem_size |
| Writer | 6+ | TensorAccessorArgs (dst) | uint32_t[] | Output buffer accessor |

**Runtime** (writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | dst_addr | uint32_t | Output buffer DRAM address |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB is bfloat16 (or float32 matching input)
- [x] DEST register holds max 8 tiles (bf16) / 4 tiles (f32) -- Wt may exceed; helpers handle sub-blocking
- [x] RM CBs count pages in sticks, tile CBs count in tiles

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(32, 32)` |
| Multi-tile | Tile iteration | `(64, 128)` |
| Non-square | W!=H | `(32, 256)` |
| Multi-batch | Batch handling | `(128, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (cb_in) | Wt | RM (tile-sized pages) | All | Row |
| c_1 (cb_eps) | 1 | Tile | Scalar [0,0] | Program |
| c_2 (cb_scaler) | 1 | Tile | Row0 | Program |
| c_3 (cb_weight) | Wt | Tile | All | Program |
| c_4 (cb_bias) | Wt | Tile | All | Program |
| c_16 (cb_tilize_out) | Wt | Tile | All | Row |
| c_17 (cb_out) | Wt | Tile | All | Row |
| c_18 (cb_untilize_out) | Wt | Tile (RM data) | All | Row |
| c_24 (cb_mean) | 1 | Tile | Col0 | Row |
| c_25 (cb_x_minus_mean) | Wt | Tile | All | Row |
| c_26 (cb_sq) | Wt | Tile | All | Row |
| c_27 (cb_var) | 1 | Tile | Col0 | Row |
| c_28 (cb_inv_std) | 1 | Tile | Col0 | Row |
| c_29 (cb_norm) | Wt | Tile | All | Row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 2 (x - mean) | SUB | All | Col0 (REDUCE_ROW output) | COL |
| 3 (square) | SQUARE | All | N/A (self) | NONE |
| 5 (norm = diff * inv_std) | MUL | All | Col0 (scalar) | COL |
| 6 (norm * gamma) | MUL | All | All (1D broadcast across rows) | NONE |
| 7 (result + beta) | ADD | All | All (1D broadcast across rows) | NONE |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader tilize + writer untilize, compute passthrough | `x` (identity) |
| 2 | subtract_mean | Phase 1 (mean) + Phase 2 (x - mean) | `x - x.mean(dim=-1, keepdim=True)` |
| 3 | full_normalize | Phases 3-5 (square, var, rsqrt, multiply) | `(x - mean) / sqrt(var + eps)` |
| 4 | affine_transform | Phases 6-7 (gamma, beta) | `F.layer_norm(x, [W], weight, bias, eps)` |

### Stage 1: data_pipeline
- **Scope**: Reader (tilize input), compute (tilize + untilize passthrough), writer (untilize output)
- **Reference**: `x` (identity passthrough)
- **Shapes**: `(32, 32)`, `(64, 128)`, `(32, 256)`, `(128, 64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes cb_in -> cb_tilize_out, copies cb_tilize_out -> cb_out (tile copy loop), untilizes cb_out -> cb_untilize_out

### Stage 2: subtract_mean
- **Scope**: Compute phases 1-2 (reduce SUM for mean, subtract mean broadcast COL)
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(32, 32)`, `(64, 128)`, `(32, 256)`, `(128, 64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Add reduce<SUM, REDUCE_ROW> with scaler=1/W for mean, then sub<COL> for x-mean

### Stage 3: full_normalize
- **Scope**: Compute phases 3-5 (square, reduce variance, add eps, rsqrt, multiply)
- **Reference**: `(x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, correction=0) + 1e-5)`
- **Shapes**: `(32, 32)`, `(64, 128)`, `(32, 256)`, `(128, 64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Add square, reduce variance, eps addition, rsqrt, normalize multiply

### Stage 4: affine_transform
- **Scope**: Compute phases 6-7 (gamma multiply, beta add), reader weight/bias loading
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight, bias, eps=1e-5)`
- **Shapes**: `(32, 32)`, `(64, 128)`, `(32, 256)`, `(128, 64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Add weight/bias CB reading, mul and add phases

### Reader Kernel
Reads RM input sticks from DRAM using TensorAccessor. For each tile-row (32 sticks), reserves cb_in for Wt tile pages, reads 32 sticks batched (tilize pattern from reference), pushes Wt pages. At program start: generates reduce scaler (1.0) in cb_scaler using `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f)`, generates eps tile in cb_eps using `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_float)`. If weight/bias provided, reads them as tiled data into cb_weight/cb_bias once at startup.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in, cb_scaler, cb_out)`

Note: The reduce scaler is 1.0 (not 1/W). We use `PoolType::SUM` with a post-reduce lambda to multiply by 1/W, keeping the standard reduce pattern. Alternative: use the scaler = 1/W directly passed from the reader. For simplicity, the reader generates scaler=1.0 and the reduce uses SUM with a post-reduce multiply-by-1/W. Actually, simplest approach: reader generates scaler = 1/W in cb_scaler via `prepare_reduce_scaler<cb_scaler>(1.0f / W)` and uses `PoolType::SUM` which applies the scaler internally during reduction. This computes mean in one step.

#### Phase 0: Tilize
```cpp
compute_kernel_lib::tilize<c_0, c_16>(Wt, 1);
```
- In: cb_in [Wt pages, pushed by reader]
- Out: cb_tilize_out [Wt tiles]

#### Phase 1: Compute Mean (reduce SUM with scaler = 1/W)
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_tilize_out, cb_scaler, cb_mean,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- A: cb_tilize_out [Wt tiles, FRESHLY PUSHED by Phase 0, persist via WaitUpfrontNoPop]
- B: cb_scaler [1 tile, program-lifetime, never popped]
- Out: cb_mean [1 tile, Col0 valid region]

#### Phase 2: x - mean (broadcast COL subtract)
```cpp
compute_kernel_lib::sub<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryOutputPolicy::Bulk>(
    cb_tilize_out, cb_mean, cb_x_minus_mean,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_tilize_out [Wt tiles, ALREADY WAITED from Phase 1, consumed and popped at end]
- B: cb_mean [1 tile, consumed and popped at end]
- Out: cb_x_minus_mean [Wt tiles, bulk pushed]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| cb_tilize_out | 0 | freed (PopAtEnd) |
| cb_mean | 0 | freed (PopAtEnd) |
| cb_x_minus_mean | Wt | freshly pushed |

#### Phase 3: Square (x - mean)^2
```cpp
compute_kernel_lib::square<
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
    compute_kernel_lib::BinaryOutputPolicy::Bulk>(
    cb_x_minus_mean, cb_sq,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_x_minus_mean [Wt tiles, persist for Phase 5]
- Out: cb_sq [Wt tiles]

#### Phase 4: Compute Variance (reduce SUM of squares with scaler = 1/W)
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
    cb_sq, cb_scaler, cb_var,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- A: cb_sq [Wt tiles, consumed and popped]
- Out: cb_var [1 tile, Col0 valid]

#### Phase 4b: Add eps and rsqrt -> inv_std
Manual phase (no single helper combines add-scalar + rsqrt):
```cpp
// Wait for var, reserve inv_std
cb_wait_front(cb_var, 1);
cb_wait_front(cb_eps, 1);
cb_reserve_back(cb_inv_std, 1);
tile_regs_acquire();
// Add eps (element-wise add with broadcast scalar)
add_bcast_scalar_init_short(cb_var, cb_eps);
add_tiles_bcast<BroadcastType::SCALAR>(cb_var, cb_eps, 0, 0, 0);
// rsqrt in-place in DST[0]
rsqrt_tile_init();
rsqrt_tile(0);
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_inv_std);
tile_regs_release();
cb_push_back(cb_inv_std, 1);
cb_pop_front(cb_var, 1);
// Note: cb_eps is NOT popped (program lifetime)
```
- Out: cb_inv_std [1 tile, Col0 valid, = 1/sqrt(var + eps)]

Alternative: use `binary_op<ADD, SCALAR>` helper for the eps addition, then a post_op lambda with rsqrt. The binary_op helper with a post_op is cleaner:
```cpp
compute_kernel_lib::add<
    compute_kernel_lib::BroadcastDim::SCALAR,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
    compute_kernel_lib::BinaryOutputPolicy::PerTile>(
    cb_var, cb_eps, cb_inv_std,
    compute_kernel_lib::BinaryInputBlockShape::single(),
    {},
    compute_kernel_lib::NoAccumulation{},
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```

#### Phase 5: Normalize (x_minus_mean * inv_std, broadcast COL)
```cpp
compute_kernel_lib::mul<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryOutputPolicy::Bulk>(
    cb_x_minus_mean, cb_inv_std, cb_norm,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_x_minus_mean [Wt tiles, ALREADY WAITED from Phase 3 NoPop, popped at end]
- B: cb_inv_std [1 tile, popped at end]
- Out: cb_norm [Wt tiles]

**CB state after Phase 5:**
| CB | Tiles | State |
|----|-------|-------|
| cb_x_minus_mean | 0 | freed |
| cb_inv_std | 0 | freed |
| cb_norm | Wt | freshly pushed |

#### Phase 6: Apply gamma (if has_weight)
```cpp
if constexpr (has_weight) {
    compute_kernel_lib::mul<
        compute_kernel_lib::BroadcastDim::NONE,
        compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
        compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::BinaryOutputPolicy::Bulk>(
        cb_norm, cb_weight, cb_out,
        compute_kernel_lib::BinaryInputBlockShape::row(Wt));
}
```
- A: cb_norm [Wt tiles, consumed]
- B: cb_weight [Wt tiles, persistent across rows, NoPop]
- Out: cb_out [Wt tiles]

#### Phase 7: Apply beta (if has_bias)
```cpp
if constexpr (has_bias) {
    compute_kernel_lib::add<
        compute_kernel_lib::BroadcastDim::NONE,
        compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
        compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
        compute_kernel_lib::BinaryOutputPolicy::Bulk>(
        cb_out, cb_bias, cb_out_final,
        compute_kernel_lib::BinaryInputBlockShape::row(Wt));
}
```
Note: When both weight and bias are present, Phase 6 output goes to cb_out, Phase 7 reads cb_out and writes to a different CB (or reuse cb_norm as scratch). When only weight or only bias or neither, the output routing adjusts. The kernel writer must handle the 4 cases (none, weight-only, bias-only, both) with compile-time flags. For the no-affine case, cb_norm data is copied directly to cb_out.

#### Phase 8: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_out, cb_untilize_out,
    compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
    compute_kernel_lib::untilize_config::WaitMode::WaitUpfront>(1);
```
- In: cb_out [Wt tiles, all pushed]
- Out: cb_untilize_out [Wt tiles as RM data]

### Writer Kernel
Follows untilize writer pattern exactly. Waits for cb_untilize_out (Wt pages per block), extracts 32 sticks per tile-row, writes each stick to DRAM via TensorAccessor. Loops Ht times.

### Critical Notes
1. **Reduce scaler = 1/W (not 1/Wt)**: The scaler must be `1.0f / W` where W is the logical width in elements. The reduce helper applies this scaler during the reduction, effectively computing mean = sum * (1/W).
2. **cb_x_minus_mean persists across Phases 3 and 5**: Phase 3 (square) uses WaitUpfrontNoPop so tiles remain for Phase 5 (normalize multiply). The square output goes to a separate cb_sq.
3. **Weight/bias are loaded once and persist**: cb_weight and cb_bias use WaitUpfrontNoPop in compute and are never popped across rows. The reader loads them once at program start.
4. **Affine output routing**: With both weight+bias, need either a scratch CB or in-place reuse. Simplest: Phase 6 writes to cb_out, Phase 7 reads cb_out and writes back to cb_norm (reuse), then untilize from cb_norm. The kernel writer should determine the optimal routing.
5. **Untilize WaitMode::WaitUpfront**: Since all Wt tiles are produced by compute before untilize runs, use WaitUpfront mode.
6. **InitUninitMode for tilize/untilize**: Since tilize and untilize are called once per row in a loop, consider InitOnly/Neither/UninitOnly pattern for rows > 1 to avoid redundant init.

### Implementation Checklist
- [ ] Reader: TensorAccessor for RM sticks, tilize batching (32 sticks -> Wt tile pages), prepare_reduce_scaler for scaler and eps, read weight/bias tiles
- [ ] Compute: 8 phases using helpers: tilize<>(), reduce<SUM, REDUCE_ROW>(), sub<COL>(), square<>(), reduce<SUM, REDUCE_ROW>(), add<SCALAR> with rsqrt post-op, mul<COL>(), mul<NONE>(), add<NONE>(), untilize<>()
- [ ] Writer: TensorAccessor for RM output sticks, untilize extraction pattern
- [ ] CB push/pop balance verified per row iteration
