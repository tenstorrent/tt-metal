# Operation Design: layer_norm

## Overview
- **Operation Name**: layer_norm
- **Category**: normalization
- **Planning Mode**: Derivative
- **Reference Operation**: moreh_group_norm (ttnn/ttnn/operations/layer_norm/moreh_group_norm_analysis.md)

## Mathematical Definition
```
E[x]      = (1/W) * sum(x[h, 0..W-1])        for each row h
Var[x]    = (1/W) * sum((x[h,w] - E[x])^2)   for each row h
output[h,w] = (x[h,w] - E[x]) / sqrt(Var[x] + eps) * gamma[w] + beta[w]
```
Layer normalization over the last dimension (W) of a 2D `[H, W]` tiled tensor. Gamma and beta are optional 1D affine parameters of shape `[1, W]`.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | (0, 1) | 1e-5 | Numerical stability constant |
| gamma | Tensor | No | shape [1, W], TILE, bfloat16 | None | Per-element scale |
| beta | Tensor | No | shape [1, W], TILE, bfloat16 | None | Per-element bias |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Rank | 4D `[N, C, H, W]` (internally 2D: Ht rows of Wt tiles) | Reshape to 4D before calling |
| Layout | TILE_LAYOUT | Must be tiled (32x32) |
| Memory | DRAM interleaved | L1/sharded not supported initially |
| Dtype | bfloat16 | Float32 future extension |
| W | Multiple of 32 | Tile-aligned width required |
| H | Multiple of 32 | Tile-aligned height required |

### Output Tensor Specification
- **Shape**: Same as input `[N, C, H, W]`
- **Dtype**: bfloat16 (same as input)
- **Layout**: TILE_LAYOUT
- **Memory**: DRAM interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile wide) | Reduce is trivial (1 tile), still uses reduce helper |
| gamma=None, beta=None | Skip affine transform, output normalized values directly |
| gamma provided, beta=None | Apply scale only |
| gamma=None, beta provided | Apply bias only |
| eps = 0 | Valid but risks division by zero for constant inputs |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | moreh_group_norm (small reader) | input_stage | Simplified: no mask generation, no gamma/beta scalar extraction. Read full row of Wt tiles, read gamma/beta as full tile rows |
| Compute | moreh_group_norm (small compute) | compute_core | Adapted: use reduce helpers for REDUCE_ROW instead of REDUCE_SCALAR, use binary_op helpers for sub/mul/add with broadcast COL and ROW |
| Writer | moreh_group_norm (writer) | output_stage | Simplified: no mean/rstd sub-tile writes, just sequential tile writes |

### Work Distribution
- **Work unit**: One tile-row = Wt tiles (all width tiles for one row of 32 pixels)
- **Grid**: Single core (1x1) for initial implementation
- **Work per core**: All Ht rows (Ht = H/32)
- **Remainder**: N/A (single core)

### Data Flow

The reader loads one full row of Wt input tiles at a time into `cb_input`. The compute kernel processes the row: (1) reduce to get mean, (2) subtract mean and square to get variance, (3) normalize, (4) optionally apply gamma/beta. The writer drains normalized output tiles to DRAM. For the "small" algorithm, all Wt tiles for a row must fit in L1 simultaneously because the input is needed three times (mean, variance, normalize).

**Three-pass compute per row (input persists in CB):**
1. **Mean pass**: REDUCE_ROW on input -> mean tile (1 tile per row, Col0 valid)
2. **Variance pass**: SUB(input, mean) broadcast COL -> centered; SQUARE(centered) -> squared; REDUCE_ROW on squared -> variance tile
3. **Normalize pass**: ADD(variance, eps) -> rsqrt -> rstd; SUB(input, mean) broadcast COL -> centered; MUL(centered, rstd) broadcast COL -> normalized; optionally MUL(normalized, gamma) broadcast ROW + ADD(result, beta) broadcast ROW -> output

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input | Input tiles (one row) | Reader | Compute | Wt | Row (persistent, 3 reads) |
| c_1 | cb_scaler | Reduce scaler (1/Wt for AVG) | Reader | Compute | 1 | Program |
| c_2 | cb_eps | Epsilon tile | Reader | Compute | 1 | Program |
| c_3 | cb_gamma | Gamma tiles (one row) | Reader | Compute | Wt | Row (if gamma) |
| c_4 | cb_beta | Beta tiles (one row) | Reader | Compute | Wt | Row (if beta) |
| c_16 | cb_out | Output tiles | Compute | Writer | Wt | Row |
| c_24 | cb_mean | E[x] per row (REDUCE_ROW output) | Compute | Compute | 1 | Row |
| c_25 | cb_centered | x - E[x] tiles | Compute | Compute | Wt | Row |
| c_26 | cb_squared | (x - E[x])^2 tiles | Compute | Compute | Wt | Row |
| c_27 | cb_var | Var[x] per row | Compute | Compute | 1 | Row |
| c_28 | cb_rstd | 1/sqrt(Var+eps) per row | Compute | Compute | 1 | Row |
| c_29 | cb_normalized | (x-mean)*rstd tiles | Compute | Compute | Wt | Row (if affine) |
| c_30 | cb_gamma_out | After gamma multiply | Compute | Compute | Wt | Row (if gamma+beta) |

**Note on cb_input persistence**: The input CB must hold all Wt tiles and NOT be popped until the row is fully processed. The compute kernel reads cb_input three times (mean, variance sub, normalize sub) using NoWaitNoPop / WaitUpfrontNoPop policies.

### Kernel Arguments

**Compile-time** (reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | gamma_has_value | uint32_t | 1 if gamma provided |
| Reader | 1 | beta_has_value | uint32_t | 1 if beta provided |
| Reader | 2+ | input_accessor_args | multiple | TensorAccessorArgs for input |
| Reader | N+ | gamma_accessor_args | multiple | TensorAccessorArgs for gamma (if present) |
| Reader | M+ | beta_accessor_args | multiple | TensorAccessorArgs for beta (if present) |

**Runtime** (reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | input_addr | uint32_t | Input tensor base address |
| Reader | 1 | gamma_addr | uint32_t | Gamma tensor base address (0 if absent) |
| Reader | 2 | beta_addr | uint32_t | Beta tensor base address (0 if absent) |
| Reader | 3 | num_rows | uint32_t | Number of tile-rows (Ht * NC) |
| Reader | 4 | Wt | uint32_t | Width in tiles |
| Reader | 5 | start_tile_id | uint32_t | Starting input tile index |

**Compile-time** (compute):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Compute | 0 | Wt | uint32_t | Width in tiles |
| Compute | 1 | num_rows | uint32_t | Number of tile-rows |
| Compute | 2 | gamma_has_value | uint32_t | 1 if gamma provided |
| Compute | 3 | beta_has_value | uint32_t | 1 if beta provided |

**Compile-time** (writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | output_accessor_args | multiple | TensorAccessorArgs for output |

**Runtime** (writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | output_addr | uint32_t | Output tensor base address |
| Writer | 1 | num_rows | uint32_t | Number of tile-rows |
| Writer | 2 | Wt | uint32_t | Width in tiles |
| Writer | 3 | start_tile_id | uint32_t | Starting output tile index |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (Wt for input/output rows, 1 for scalars)
- [x] Reduce scaler CB is bfloat16 (cb_scaler c_1)
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- helpers manage this automatically
- [x] RM CBs count pages in sticks, tile CBs count in tiles -- all CBs are tile-based here
- [x] Input CB persists across 3 compute passes -- use NoPop policies

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` with rtol/atol per stage

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tests reduce across W | `(1, 1, 32, 128)` |
| Multi-tile H | Tests row iteration | `(1, 1, 128, 64)` |
| Non-square | W >> H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(2, 1, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (input) | Wt | TILE | All | Row (persistent, no pop until row done) |
| c_1 (scaler) | 1 | TILE (bfloat16) | Row0 (reduce scaler format) | Program |
| c_2 (eps) | 1 | TILE | Scalar [0,0] | Program |
| c_3 (gamma) | Wt | TILE | Row0 (1D broadcast) | Row |
| c_4 (beta) | Wt | TILE | Row0 (1D broadcast) | Row |
| c_16 (out) | Wt | TILE | All | Row |
| c_24 (mean) | 1 | TILE | Col0 (REDUCE_ROW output) | Row |
| c_25 (centered) | Wt | TILE | All | Row |
| c_26 (squared) | Wt | TILE | All | Row |
| c_27 (var) | 1 | TILE | Col0 (REDUCE_ROW output) | Row |
| c_28 (rstd) | 1 | TILE | Col0 (after rsqrt of var+eps) | Row |
| c_29 (normalized) | Wt | TILE | All | Row (if affine) |
| c_30 (gamma_out) | Wt | TILE | All | Row (if gamma+beta) |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| Sub mean | SUB | All (cb_input) | Col0 (cb_mean) | COL |
| Mul rstd | MUL | All (cb_centered) | Col0 (cb_rstd) | COL |
| Mul gamma | MUL | All (cb_normalized) | Row0 (cb_gamma, 1D [1,W]) | ROW |
| Add beta | ADD | All (cb_gamma_out) | Row0 (cb_beta, 1D [1,W]) | ROW |
| Add eps | ADD | Col0 (cb_var) | Scalar (cb_eps) | SCALAR |

**Validation**: REDUCE_ROW output -> Col0 valid. Gamma/beta are [1, W] -> Row0 valid. All broadcasts match their operand valid regions.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_passthrough | Reader + Writer pipeline, compute copies input to output | `input_tensor` (identity) |
| 2 | mean_reduction | Compute: REDUCE_ROW for mean, then SUB mean from input | `x - x.mean(dim=-1, keepdim=True)` |
| 3 | variance_normalize | Compute: variance + rsqrt + full normalization (no affine) | `F.layer_norm(x, [W])` without gamma/beta |
| 4 | affine_transform | Compute: gamma * normalized + beta | `F.layer_norm(x, [W], weight=gamma, bias=beta)` |

### Stage 1: data_passthrough
- **Scope**: reader kernel, writer kernel, compute kernel (copy only)
- **Reference**: `input_tensor` (passthrough identity)
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,128,64)`, `(1,1,32,256)`, `(2,1,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute does `cb_wait_front(cb_input, Wt)` then copies each tile via `copy_tile` to `cb_out`, then `cb_pop_front(cb_input, Wt)`.

### Stage 2: mean_reduction
- **Scope**: compute kernel adds REDUCE_ROW + SUB broadcast COL
- **Reference**: `torch_input - torch_input.to(torch.float32).mean(dim=-1, keepdim=True).to(torch.bfloat16)`
- **Delta from previous**: Compute now reduces input row to mean (REDUCE_ROW), then subtracts mean from each input tile using broadcast COL. Output is centered data.
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,128,64)`, `(1,1,32,256)`, `(2,1,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1

### Stage 3: variance_normalize
- **Scope**: compute kernel adds variance computation, rsqrt, full normalization
- **Reference**: `torch.nn.functional.layer_norm(torch_input.to(torch.float32), [torch_input.shape[-1]]).to(torch.bfloat16)`
- **Delta from previous**: After centering, compute squares centered tiles, reduces to variance, adds eps, computes rsqrt, multiplies centered by rstd.
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,128,64)`, `(1,1,32,256)`, `(2,1,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Stage 4: affine_transform
- **Scope**: reader adds gamma/beta loading, compute adds gamma multiply + beta add
- **Reference**: `torch.nn.functional.layer_norm(torch_input.to(torch.float32), [torch_input.shape[-1]], weight=gamma.to(torch.float32).squeeze(0), bias=beta.to(torch.float32).squeeze(0)).to(torch.bfloat16)`
- **Delta from previous**: Reader loads gamma/beta row tiles. Compute multiplies normalized by gamma (ROW broadcast) and adds beta (ROW broadcast).
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,128,64)`, `(1,1,32,256)`, `(2,1,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel

Per-row loop: `cb_reserve_back(cb_input, Wt)`, read Wt tiles from DRAM via TensorAccessor, `cb_push_back(cb_input, Wt)`. For stage 4, also load gamma/beta rows into cb_gamma/cb_beta similarly. Scaler and eps tiles are filled once at program start using `dataflow_kernel_lib::prepare_reduce_scaler` and a manual fill for eps.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input, cb_scaler, cb_out);`

The compute kernel needs three-argument startup because it uses reduce (which reads from cb_scaler as srcB).

#### Phase 1: Mean (REDUCE_ROW)
```cpp
// Input already waited by WaitUpfrontNoPop
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_input, cb_scaler, cb_mean,
    compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));
```
Produces 1 tile per row in cb_mean with valid data in Col0.

#### Phase 2: Subtract mean (x - E[x])
```cpp
compute_kernel_lib::sub<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,   // input already in CB
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    cb_input, cb_mean, cb_centered,
    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
```
**CRITICAL**: cb_input uses NoWaitNoPop (persistent), cb_mean uses WaitAndPopPerTile (1 tile, COL broadcast pops once per row).

#### Phase 3: Square centered values
```cpp
compute_kernel_lib::square<
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    cb_centered, cb_squared,
    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
```

#### Phase 4: Reduce squared to variance (REDUCE_ROW)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    cb_squared, cb_scaler, cb_var,
    compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));
```

#### Phase 5: var + eps, then rsqrt
```cpp
// ADD eps to var (SCALAR broadcast, 1x1 shape)
compute_kernel_lib::add<
    compute_kernel_lib::BroadcastDim::SCALAR>(
    cb_var, cb_eps, cb_var_eps,  // use a temp or in-place
    compute_kernel_lib::BinaryInputBlockShape::single());
```
Then rsqrt via post_reduce_op or manual:
```cpp
// Manual: cb_wait_front(cb_var_eps, 1); tile_regs_acquire(); copy_tile(cb_var_eps, 0, 0);
// rsqrt_tile_init(); rsqrt_tile(0); pack to cb_rstd; tile_regs_release(); cb_pop_front
```

**Note**: Since there is no helper for standalone rsqrt, this phase uses raw compute API. The `add` helper handles the eps addition. Rsqrt is a single SFPU op on one tile.

#### Phase 6: Multiply centered by rstd (COL broadcast)
```cpp
// Re-compute centered: SUB input - mean again (input still in CB from persistent hold)
// OR: keep cb_centered from phase 2 if it hasn't been popped
// Decision: Phase 3 square uses WaitUpfrontPopAtEnd on cb_centered, so centered IS consumed.
// We must recompute: sub(cb_input, cb_mean_reloaded, cb_centered2, ...)
// BUT cb_mean was popped in phase 2.
```

**REVISED APPROACH**: To avoid recomputation, we restructure the data flow:

1. **Mean**: reduce input -> cb_mean (persist, don't pop)
2. **Sub mean**: sub(input, mean) -> cb_centered (persist both inputs)
3. **Square**: square(centered) -> cb_squared (centered persists via NoWaitNoPop)
4. **Var reduce**: reduce squared -> cb_var
5. **Rsqrt**: var + eps -> rsqrt -> cb_rstd
6. **Normalize**: mul(centered, rstd) -> cb_normalized or cb_out

This requires cb_centered to persist through phases 3-6. We use NoWaitNoPop on cb_centered in the square phase, keeping tiles for phase 6.

**Revised Phase 2 (sub mean)**:
```cpp
compute_kernel_lib::sub<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,   // input persistent
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryOutputPolicy::Bulk>(        // centered persists
    cb_input, cb_mean, cb_centered,
    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
```

**Revised Phase 3 (square)**:
```cpp
compute_kernel_lib::square<
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,   // centered persistent
    compute_kernel_lib::BinaryOutputPolicy::PerTile>(
    cb_centered, cb_squared,
    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
```

**Revised Phase 6 (normalize = centered * rstd)**:
```cpp
compute_kernel_lib::mul<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,     // centered still in CB
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    cb_centered, cb_rstd, cb_out_or_normalized,
    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
// NOW pop cb_centered manually: cb_pop_front(cb_centered, Wt)
// AND pop cb_input: cb_pop_front(cb_input, Wt)
```

#### Phase 7 (if gamma): Multiply by gamma (ROW broadcast)
```cpp
compute_kernel_lib::mul<
    compute_kernel_lib::BroadcastDim::ROW,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    cb_normalized, cb_gamma, cb_gamma_out_or_cb_out,
    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
```

#### Phase 8 (if beta): Add beta (ROW broadcast)
```cpp
compute_kernel_lib::add<
    compute_kernel_lib::BroadcastDim::ROW,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    cb_gamma_out, cb_beta, cb_out,
    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
```

### Writer Kernel

Per-row loop: `cb_wait_front(cb_out, Wt)`, write Wt tiles to DRAM via TensorAccessor, `cb_pop_front(cb_out, Wt)`.

### Critical Notes

1. **cb_input persistence**: Input tiles are read ONCE by the reader but used THREE times by compute (mean reduce, sub mean, normalize mul). Use WaitUpfrontNoPop / NoWaitNoPop policies. Manual `cb_pop_front(cb_input, Wt)` at end of row.

2. **cb_centered persistence**: Centered tiles are used TWICE (square for variance, multiply by rstd for normalize). Same NoPop pattern, manual pop after phase 6.

3. **cb_mean persistence**: Mean is used ONCE in sub (phase 2). With COL broadcast and WaitAndPopPerTile for input_b, the helper pops it once per row automatically.

4. **Reduce scaler for REDUCE_ROW with SUM**: Use `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, Wt>()`. With SUM type, scaler = 1.0. Then the mean is sum/Wt. **Actually**: use AVG type so `scaler = 1/Wt`, producing mean directly. Use `calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::AVG, ReduceDim::REDUCE_ROW, Wt>()`. But Wt must be a compile-time constant for the template. If Wt is runtime, use `prepare_reduce_scaler<cb_scaler>(1.0f / Wt)`.

5. **Manual rsqrt**: No helper wraps standalone rsqrt. Must use raw API: `rsqrt_tile_init()`, `rsqrt_tile(dst_idx)`.

6. **CB page count for persistent CBs**: cb_input and cb_centered both hold Wt pages. For large W (e.g., W=8192, Wt=256), this requires 256 tiles * 2KB = 512KB per CB. With two persistent CBs (input + centered) that's 1MB. This is close to L1 limit (1.5MB). For very wide tensors, a "large" algorithm with re-reads would be needed. The initial implementation targets the "small" algorithm where all tiles fit.

7. **Reduce scaler value**: For REDUCE_ROW with AVG: `scaler = 1/Wt` (the reduce helper with AVG type handles the per-tile-width factor internally via the LLK). The `calculate_and_prepare_reduce_scaler` template with `reduce_volume = Wt` computes `1/Wt` for REDUCE_ROW.

### Implementation Checklist
- [ ] Reader: TensorAccessor reads for input/gamma/beta, prepare_reduce_scaler for cb_scaler, manual fill for cb_eps
- [ ] Compute: 6-8 phases using helpers: reduce<SUM/AVG, REDUCE_ROW>, sub<COL>, square, reduce<SUM/AVG, REDUCE_ROW>, add<SCALAR>, mul<COL>, mul<ROW>, add<ROW>. Manual rsqrt.
- [ ] Writer: TensorAccessor writes for output
- [ ] CB push/pop balance verified: input and centered use manual pops after persistent use
- [ ] Scaler CB filled with correct value for AVG REDUCE_ROW
