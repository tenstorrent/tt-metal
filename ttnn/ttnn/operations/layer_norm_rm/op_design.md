# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), reduce_w (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[b,h] = sum(x[b,h,:]) / W
centered[b,h,w] = x[b,h,w] - mean[b,h]
var[b,h] = sum(centered[b,h,:]^2) / W
output[b,h,w] = centered[b,h,w] * rsqrt(var[b,h] + epsilon)
output[b,h,w] = output[b,h,w] * gamma[w] + beta[w]   (if affine)
```
Row-wise layer normalization on RM interleaved tensors. Normalizes across the last dimension (W).

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Stability constant added to variance |
| gamma | Tensor | No | shape (1,1,1,W) | None | Per-element scale (bfloat16 RM) |
| beta | Tensor | No | shape (1,1,1,W) | None | Per-element bias (bfloat16 RM) |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Layout | ROW_MAJOR | "Input must be row-major" |
| Memory | Interleaved | "Input must be interleaved" |
| Dtype | bfloat16 | "Input must be bfloat16" |
| Last 2 dims | Tile-aligned (multiples of 32) | "H and W must be multiples of 32" |
| Rank | >= 2 | "Input must be at least 2D" |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: bfloat16
- **Layout**: ROW_MAJOR
- **Memory**: Interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| gamma=None, beta=None | Pure normalization (no affine) |
| gamma provided, beta=None | Scale only: output = normalized * gamma |
| W = 32 (single tile width) | Wt=1, reduce degenerates to single-tile |
| All-zero row | Outputs zeros (0/rsqrt(0+eps) ~ 0) |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize_analysis | input_stage | Read input RM sticks + optional gamma/beta RM sticks |
| Compute | reduce_w_analysis + new | compute_core | Tilize, multi-phase layernorm, untilize |
| Writer | untilize_analysis | output_stage | Write RM sticks from untilized output CB |

### Work Distribution
- **Work unit**: tile-row block (32 sticks spanning full width = Wt tiles)
- **Grid**: 1D, up to `grid_size.x * grid_size.y` cores
- **Work per core**: `nblocks_per_core` tile-rows via `split_blocks_for_tilize`
- **Remainder**: Cliff core gets `nblocks % nblocks_per_core` blocks

### Data Flow
Reader reads 32 RM input sticks per block into c_0. Compute tilizes c_0 into c_24, performs layernorm phases in tile domain (reduce_mean, subtract_mean, square, reduce_var, mul_inv_std, optional affine), untilizes result into c_16. Writer drains c_16 as RM sticks to DRAM. Gamma/beta are read once, tilized, and persist.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input_rm | Input RM sticks (tile-sized pages) | Reader | Compute(tilize) | Wt | Block |
| c_1 | cb_tilized | Tilized input tiles | Compute(tilize) | Compute(sub,square) | Wt | Block (persists for reuse) |
| c_2 | cb_reduce_scaler | Reduce scaler tile (1/W) | Reader | Compute(reduce) | 1 | Program |
| c_3 | cb_mean | Mean per row (reduced) | Compute(reduce) | Compute(sub) | 1 | Row |
| c_4 | cb_centered | Centered tiles (x - mean) | Compute(sub) | Compute(square,mul) | Wt | Block (persists for reuse) |
| c_5 | cb_squared | Squared centered tiles | Compute(square) | Compute(reduce_var) | Wt | Block |
| c_6 | cb_inv_std | inv_std = rsqrt(var + eps) per row | Compute(reduce+rsqrt) | Compute(mul) | 1 | Row |
| c_7 | cb_eps | Epsilon tile (constant) | Reader | Compute(add_eps) | 1 | Program |
| c_16 | cb_output_rm | Output RM sticks (tile-sized pages) | Compute(untilize) | Writer | Wt | Block |
| c_24 | cb_normalized | Normalized tiles (before affine) | Compute(mul) | Compute(affine or untilize) | Wt | Block |
| c_25 | cb_gamma_tilized | Tilized gamma tiles | Compute(tilize) | Compute(mul_gamma) | Wt | Program (persistent) |
| c_26 | cb_beta_tilized | Tilized beta tiles | Compute(tilize) | Compute(add_beta) | Wt | Program (persistent) |
| c_27 | cb_gamma_rm | Gamma RM sticks (tile-sized pages) | Reader | Compute(tilize_gamma) | Wt | Once |
| c_28 | cb_beta_rm | Beta RM sticks (tile-sized pages) | Reader | Compute(tilize_beta) | Wt | Once |
| c_29 | cb_affine_out | Output after affine transform | Compute(affine) | Compute(untilize) | Wt | Block |

### Kernel Arguments

**Compile-time (Reader)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Input stick bytes (W * 2) |
| 1 | has_gamma | uint32_t | 1 if gamma provided, 0 otherwise |
| 2 | has_beta | uint32_t | 1 if beta provided, 0 otherwise |
| 3+ | input_accessor_args | uint32_t[] | TensorAccessorArgs for input |
| ... | gamma_accessor_args | uint32_t[] | TensorAccessorArgs for gamma (if present) |
| ... | beta_accessor_args | uint32_t[] | TensorAccessorArgs for beta (if present) |

**Runtime (Reader)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | num_sticks | uint32_t | Total sticks for this core (nblocks * 32) |
| 2 | Wt | uint32_t | Tiles per row (ntiles_per_block) |
| 3 | start_stick_id | uint32_t | First stick ID for this core |
| 4 | gamma_addr | uint32_t | Gamma buffer address (0 if none) |
| 5 | beta_addr | uint32_t | Beta buffer address (0 if none) |

**Compile-time (Compute)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | nblocks_per_core | uint32_t | Number of tile-row blocks this core processes |
| 1 | Wt | uint32_t | Tiles per row (width dimension) |
| 2 | has_gamma | uint32_t | 1 if gamma present |
| 3 | has_beta | uint32_t | 1 if beta present |

**Runtime (Compute)**: None (all info in compile-time args).

**Compile-time (Writer)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Output stick bytes (W * 2) |
| 1 | Wt | uint32_t | Tiles per row |
| 2+ | output_accessor_args | uint32_t[] | TensorAccessorArgs for output |

**Runtime (Writer)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_blocks | uint32_t | Number of tile-row blocks for this core |
| 2 | start_stick_id | uint32_t | First output stick ID |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB (c_2) is bfloat16 (Float16_b)
- [x] DEST register holds max 8 tiles (bf16) / 4 tiles (f32)
- [x] RM CBs count pages in tiles (tile-sized pages for tilize compatibility)
- [x] Epsilon CB (c_7) is bfloat16 (Float16_b)

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` (rtol/atol per stage)
- Test shapes:

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration on W | `(1, 1, 32, 128)` |
| Multi-tile HW | Tile iteration on both | `(1, 1, 64, 128)` |
| Non-square | W != H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (input_rm) | Wt | Tile-sized | All (RM sticks) | Block |
| c_1 (tilized) | Wt | Tile | All | Block (persists for sub phase) |
| c_2 (reduce_scaler) | 1 | Tile | Row0 of each face | Program |
| c_3 (mean) | 1 | Tile | Col0 only | Row |
| c_4 (centered) | Wt | Tile | All | Block (persists for mul_inv_std) |
| c_5 (squared) | Wt | Tile | All | Block |
| c_6 (inv_std) | 1 | Tile | Col0 only | Row |
| c_7 (eps) | 1 | Tile | Scalar ([0,0]) | Program |
| c_16 (output_rm) | Wt | Tile-sized | All (RM sticks) | Block |
| c_24 (normalized) | Wt | Tile | All | Block |
| c_25 (gamma_tilized) | Wt | Tile | Row0 only | Program |
| c_26 (beta_tilized) | Wt | Tile | Row0 only | Program |
| c_27 (gamma_rm) | Wt | Tile-sized | Row0 sticks | Once |
| c_28 (beta_rm) | Wt | Tile-sized | Row0 sticks | Once |
| c_29 (affine_out) | Wt | Tile | All | Block |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| subtract_mean | SUB | c_1: All | c_3: Col0 | COL |
| square | SQUARE | c_4: All | c_4: All | NONE |
| add_eps | ADD | c_5(var): Col0 | c_7: Scalar | SCALAR |
| mul_inv_std | MUL | c_4: All | c_6: Col0 | COL |
| mul_gamma | MUL | c_24: All | c_25: Row0 | ROW |
| add_beta | ADD | c_29(scaled): All | c_26: Row0 | ROW |

Note: reduce(SUM, REDUCE_ROW) output -> Col0 valid. gamma/beta are 1D (1,1,1,W) -> after tilize Row0 valid. Epsilon is scalar -> [0,0] valid.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader(input) + tilize + untilize + Writer | Identity passthrough |
| 2 | reduce_mean | + reduce(SUM, REDUCE_ROW) with scaler=1/W | Row means (shape [B,C,H,1]) |
| 3 | subtract_mean | + sub<COL> (x - mean) | Centered (zero-mean rows) |
| 4 | variance_inv_std | + square + reduce_var + add_eps + rsqrt | Full layer_norm (no affine) |
| 5 | affine | + gamma/beta tilize + mul<ROW> + add<ROW> | Full layer_norm with affine |

### Stage 1: data_pipeline
- **Scope**: Reader kernel (input sticks), compute (tilize c_0->c_1, untilize c_1->c_16), writer kernel (output sticks)
- **Reference**: `x` (identity passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes input then immediately untilizes to output. No normalization phases.

### Stage 2: reduce_mean
- **Scope**: Compute adds reduce(SUM, REDUCE_ROW) phase with scaler=1/W to produce mean
- **Reference**: `x.mean(dim=-1, keepdim=True).expand_as(x)` (broadcast mean to full shape for comparison via untilize of mean broadcast)
- **Actually**: Output is `x.mean(dim=-1, keepdim=True)` with output shape `(B,C,H,1)` padded to tile alignment `(B,C,H,32)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Adds reduce scaler setup in reader, reduce phase in compute, output is reduced (1 tile per row) instead of full-width. Writer output is tile layout (reduced tiles).

### Stage 3: subtract_mean
- **Scope**: Compute adds sub<COL> phase: centered = x - mean
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Adds subtraction phase, output returns to full width (untilized centered output)

### Stage 4: variance_inv_std
- **Scope**: Compute adds square + reduce_var + add_eps + rsqrt + mul_inv_std phases
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=None, bias=None, eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Adds 5 compute phases (square, reduce_var, add_eps via rsqrt post-op, mul_inv_std), epsilon CB setup in reader

### Stage 5: affine
- **Scope**: Reader adds gamma/beta stick reads + compute adds tilize for gamma/beta + mul<ROW> + add<ROW>
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma.squeeze(), bias=beta.squeeze(), eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Adds gamma/beta read, tilize, and affine multiply+add phases

### Reader Kernel
Reads 32 RM input sticks per block into c_0 using TensorAccessor (same pattern as tilize reader). Generates reduce scaler tile into c_2 using `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<c_2, SUM, REDUCE_ROW, 32, W>()`. Generates epsilon tile into c_7 using `dataflow_kernel_lib::prepare_reduce_scaler<c_7>(eps_float)` with scalar fill pattern. Optionally reads gamma/beta sticks (32 sticks from 1-row tensor, only row 0 has data) into c_27/c_28.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm)`
Note: 3-arg form sets srcA=c_0, srcB=c_2, ocb=c_16.

**Optional gamma/beta tilize** (once at kernel start, before main loop):
```cpp
// If has_gamma:
compute_kernel_lib::tilize<c_27, c_25>(Wt, 1);  // tilize gamma RM -> tile
// If has_beta:
compute_kernel_lib::tilize<c_28, c_26>(Wt, 1);  // tilize beta RM -> tile
```

**Main loop** (per tile-row block, nblocks_per_core iterations):

#### Phase 1: Tilize input
```cpp
compute_kernel_lib::tilize<c_0, c_1>(Wt, 1);
```
- In: c_0 [Wt pages, pushed by reader]
- Out: c_1 [Wt tiles, pushed by tilize]

#### Phase 2: Reduce mean
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_1, c_2, c_3,
    ReduceInputBlockShape::of(1, Wt, 1));
```
- A: c_1 [Wt tiles, FRESHLY PUSHED by Phase 1, NoPop — persists for Phase 3]
- Scaler: c_2 [1 tile, persistent from reader, holds 1/W]
- Out: c_3 [1 tile, Col0 valid — row mean]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | waited, not popped — persists for Phase 3 |
| c_3 | 1 | freshly pushed (mean) |

#### Phase 3: Subtract mean (centered = x - mean)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_1, c_3, c_4,
    BinaryInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop]
- B: c_3 [1 tile, WaitAndPopPerTile — COL broadcast waits 1 tile, pops after row]
- Out: c_4 [Wt tiles]

Manual pop after Phase 3:
```cpp
cb_pop_front(c_1, Wt);  // Free tilized input — no longer needed
```

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | 0 | freed |
| c_3 | 0 | freed by sub's COL pop |
| c_4 | Wt | freshly pushed (centered) |

#### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_4, c_5, BinaryInputBlockShape::of(1, Wt));
```
- In: c_4 [Wt tiles, FRESHLY PUSHED by Phase 3, WaitUpfrontNoPop — persists for Phase 7]
- Out: c_5 [Wt tiles]

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| c_4 | Wt | waited, not popped — persists for Phase 7 |
| c_5 | Wt | freshly pushed (squared) |

#### Phase 5: Reduce variance + rsqrt (via post_reduce_op)
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_5, c_2, c_6,
    ReduceInputBlockShape::of(1, Wt, 1),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst_idx) {
        // Add epsilon and rsqrt in-place in DEST
        // Epsilon tile is pre-loaded in c_7
        reduce_uninit<false>();  // clear packer mask before binary op
        cb_wait_front(c_7, 1);
        add_tiles_init(c_6, c_7);  // reconfigure for add
        add_tiles(c_6, c_7, dst_idx, 0, dst_idx);  // var + eps (c_6 not actually used as CB here, DEST operand)
        // Actually: add_tiles reads from DEST[dst_idx] and CB[c_7, tile 0]
        // Correction: We cannot add_tiles from DEST. Need different approach.
    });
```

**REVISED Phase 5 approach**: The post_reduce_op runs while result is in DEST, but `add_tiles` requires both operands in CBs. Instead, we split into: reduce -> pack to temp -> add_eps -> rsqrt -> pack to c_6.

**Actually, the simplest approach**: Use the reduce with scaler=1/W to get mean(centered^2) = variance. Then pack variance to a temp CB, add epsilon via binary add<SCALAR>, apply rsqrt via post_op on the add.

Revised Phase 5+6: Reduce variance, then add_eps+rsqrt separately.

#### Phase 5: Reduce variance
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_5, c_2, c_6,
    ReduceInputBlockShape::of(1, Wt, 1));
```
- In: c_5 [Wt tiles, WaitAndPopPerTile — consumed tile by tile]
- Scaler: c_2 [1 tile, persistent, 1/W]
- Out: c_6 [1 tile, Col0 valid — variance]

#### Phase 6: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_6, c_7, c_6,
    BinaryInputBlockShape::of(1, 1),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_6 [1 tile, variance, consumed]
- B: c_7 [1 tile, epsilon scalar, persistent NoPop]
- Out: c_6 [1 tile, inv_std = rsqrt(var + eps)]

Note: c_6 is used as both input and output here. The add helper will wait+pop c_6 (the variance), then reserve+push to c_6 (the inv_std). This works because WaitAndPopPerTile pops before reserve.

#### Phase 7: Multiply by inv_std (normalized = centered * inv_std)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_4, c_6, c_24,
    BinaryInputBlockShape::of(1, Wt));
```
- A: c_4 [Wt tiles, ALREADY WAITED from Phase 4, NoWaitNoPop]
- B: c_6 [1 tile, inv_std, WaitAndPopPerTile — COL broadcast]
- Out: c_24 [Wt tiles, normalized]

Manual pop after Phase 7:
```cpp
cb_pop_front(c_4, Wt);  // Free centered — no longer needed
```

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_4 | 0 | freed |
| c_6 | 0 | freed by mul's COL pop |
| c_24 | Wt | freshly pushed (normalized) |

#### Phase 8 (conditional): Multiply by gamma
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::WaitUpfrontNoPop,
        BinaryOutputPolicy::PerTile,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        c_24, c_25, c_29,
        BinaryInputBlockShape::of(1, Wt));
}
```
- A: c_24 [Wt tiles, normalized, consumed]
- B: c_25 [Wt tiles, gamma tilized, persistent NoPop]
- Out: c_29 [Wt tiles, scaled]

#### Phase 9 (conditional): Add beta
```cpp
if constexpr (has_beta) {
    uint32_t src_cb = has_gamma ? c_29 : c_24;
    compute_kernel_lib::add<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::WaitUpfrontNoPop,
        BinaryOutputPolicy::PerTile,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        src_cb, c_26, c_29,
        BinaryInputBlockShape::of(1, Wt));
}
```
- A: src_cb [Wt tiles, consumed]
- B: c_26 [Wt tiles, beta tilized, persistent NoPop]
- Out: c_29 [Wt tiles, affine output]

#### Phase 10: Untilize
```cpp
uint32_t final_cb = (has_gamma || has_beta) ? c_29 : c_24;
compute_kernel_lib::untilize<Wt, final_cb, c_16>(1);
```
- In: final_cb [Wt tiles]
- Out: c_16 [Wt tile-sized pages of RM sticks]

### Writer Kernel
Waits for Wt pages in c_16, extracts 32 RM sticks, writes each to DRAM via TensorAccessor. Same pattern as untilize writer: `get_read_ptr(c_16)` + stride through 32 rows, `noc_async_write` each stick, barrier, `cb_pop_front(c_16, Wt)`.

### Critical Notes
1. **c_1 persists across Phase 2-3**: Tilized input must remain available for subtract_mean. Use WaitUpfrontNoPop in reduce, NoWaitNoPop in sub, then manual pop.
2. **c_4 persists across Phase 4-7**: Centered values needed for both square (Phase 4) and mul_inv_std (Phase 7). Use WaitUpfrontNoPop in square, NoWaitNoPop in mul, then manual pop.
3. **c_6 reuse as input and output in Phase 6**: Works because WaitAndPopPerTile pops before the output reserve/push cycle.
4. **Gamma/beta are ROW broadcast**: They are 1D tensors (1,1,1,W) tilized into Wt tiles. After tilize only Row0 of each tile is valid. ROW broadcast replicates this across all rows.
5. **Reduce scaler**: Uses `calculate_and_prepare_reduce_scaler<c_2, SUM, REDUCE_ROW, 32, W>()` which computes 1/W for AVG-like behavior. Since we use PoolType::SUM with scaler=1/W, this effectively computes the mean.
6. **Epsilon tile format**: Epsilon is a scalar broadcast. Use `prepare_reduce_scaler<c_7>(eps)` to fill a tile compatible with SCALAR broadcast binary add. Actually, for binary add<SCALAR> the B operand needs all elements filled (not just row 0). Use a simple fill pattern: reserve c_7, zero tile, fill all elements with eps, push. The binary_op SCALAR broadcast reads tile[0,0] and broadcasts it.
7. **InitUninitMode for back-to-back tilize calls**: When tilizing gamma and beta before the main loop, use InitAndUninit for each since they operate on different CB pairs. The main loop tilize also uses InitAndUninit since there are other operations between iterations.

### Implementation Checklist
- [ ] Reader: TensorAccessor for input sticks, reduce scaler via calculate_and_prepare_reduce_scaler, epsilon tile generation, optional gamma/beta stick reads
- [ ] Compute: 10 phases using helpers: tilize, reduce(SUM, REDUCE_ROW), sub(COL), square, reduce(SUM, REDUCE_ROW), add(SCALAR)+rsqrt post-op, mul(COL), mul(ROW), add(ROW), untilize
- [ ] Writer: TensorAccessor for output sticks, 32 sticks per block
- [ ] CB push/pop balance verified — manual pops for c_1 (after Phase 3) and c_4 (after Phase 7)
