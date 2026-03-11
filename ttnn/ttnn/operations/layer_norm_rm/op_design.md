# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), batch_norm (compute_core)

## Mathematical Definition
```
mean[b,h] = sum(x[b,h,:]) / W
centered[b,h,w] = x[b,h,w] - mean[b,h]
var[b,h] = sum(centered[b,h,:]^2) / W
output[b,h,w] = centered[b,h,w] * rsqrt(var[b,h] + epsilon)
if gamma: output[b,h,w] *= gamma[w]
if beta:  output[b,h,w] += beta[w]
```
Layer normalization over the last dimension (W) of a row-major interleaved tensor. Statistics are computed per-row.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant |
| gamma | Tensor | No | shape (1,1,1,W) | None | Scale parameter |
| beta | Tensor | No | shape (1,1,1,W) | None | Shift parameter |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | BFLOAT16 | "Input must be bfloat16" |
| layout | ROW_MAJOR_LAYOUT | "Input must be row-major" |
| memory | INTERLEAVED | "Input must be interleaved" |
| rank | >= 2 | "Input must be at least 2D" |
| H % 32 | == 0 | "Height must be aligned to 32" |
| W % 32 | == 0 | "Width must be aligned to 32" |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR_LAYOUT
- **Memory**: INTERLEAVED

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce_row produces 1 tile per row |
| gamma provided, beta None | Apply scale only, skip bias add |
| gamma None, beta provided | Skip scale, apply bias add only |
| Very small epsilon | Numerical precision limited by bfloat16 |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader (RM sticks) | tilize analysis | input_stage | 3 read passes per tile-row; gamma/beta stick duplication |
| Compute (tilize) | tilize analysis | input_stage | Multiple tilize calls with different output CBs |
| Compute (normalize) | batch_norm analysis | compute_core | Replace channel-group pattern with row-wise reduce+binary ops using helpers |
| Compute (untilize) | untilize analysis | output_stage | Single untilize call per tile-row |
| Writer (RM sticks) | untilize analysis | output_stage | Stick extraction via get_read_ptr, TensorAccessor writes |

### Work Distribution
- **Work unit**: tile-row (32 sticks x full width = Wt tiles)
- **Grid**: 1D linear (from `split_blocks_for_tilize`)
- **Work per core**: `nblocks_per_core` tile-rows (cliff core gets remainder)
- **Remainder**: Last core is cliff core with `nblocks_per_core_cliff` tile-rows

### Data Flow
Three-pass architecture per tile-row. Each pass reads the same RM input sticks from DRAM and tilizes in-kernel. Pass 1 computes mean via row reduction. Pass 2 subtracts mean, squares, and reduces to get variance. Pass 3 normalizes using mean and inv_std, optionally applies gamma/beta, then untilizes and writes RM output to DRAM.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_rm_in | RM input staging for tilize | Reader | Compute (tilize) | Wt | Block (per tilize) |
| c_1 | cb_tilized | Tilized input tiles | Compute (tilize) | Compute (reduce/binary) | Wt | Block (per pass) |
| c_2 | cb_scaler | Reduce scaler tile (1/W, bf16) | Reader | Compute (reduce) | 1 | Program |
| c_3 | cb_eps | Epsilon scalar tile (bf16) | Reader | Compute (add) | 1 | Program |
| c_4 | cb_gamma | Gamma tilized (optional) | Compute (tilize) | Compute (mul) | Wt | Program |
| c_5 | cb_beta | Beta tilized (optional) | Compute (tilize) | Compute (add) | Wt | Program |
| c_24 | cb_mean | Row-wise mean (1 tile) | Compute (reduce) | Compute (sub) | 1 | Tile-row |
| c_25 | cb_inter | Intermediate tiles | Compute (binary) | Compute (binary/reduce) | Wt | Phase |
| c_26 | cb_var | Row-wise variance (1 tile) | Compute (reduce) | Compute (add+rsqrt) | 1 | Tile-row |
| c_27 | cb_inv_std | 1/sqrt(var+eps) (1 tile) | Compute (add+rsqrt) | Compute (mul) | 1 | Tile-row |
| c_16 | cb_out_pre | Final tiles before untilize | Compute (binary) | Compute (untilize) | Wt | Block |
| c_17 | cb_out_rm | Untilized RM output | Compute (untilize) | Writer | Wt | Block |

### Kernel Arguments

**Compile-time** (Reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | Width in bytes: W * 2 |
| Reader | 1 | has_gamma | uint32_t | 1 if gamma provided |
| Reader | 2 | has_beta | uint32_t | 1 if beta provided |
| Reader | 3+ | TensorAccessorArgs | uint32_t[] | Input buffer accessor |

**Compile-time** (Compute):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Compute | 0 | Wt | uint32_t | Width in tiles (W/32) |
| Compute | 1 | num_tile_rows | uint32_t | Tile-rows this core processes |
| Compute | 2 | has_gamma | uint32_t | 1 if gamma provided |
| Compute | 3 | has_beta | uint32_t | 1 if beta provided |

**Compile-time** (Writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | stick_size | uint32_t | Width in bytes: W * 2 |
| Writer | 1 | Wt | uint32_t | Width in tiles |
| Writer | 2+ | TensorAccessorArgs | uint32_t[] | Output buffer accessor |

**Runtime** (Reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer DRAM address |
| Reader | 1 | num_tile_rows | uint32_t | Tile-rows for this core |
| Reader | 2 | start_stick_id | uint32_t | First stick ID for this core |
| Reader | 3 | gamma_addr | uint32_t | Gamma buffer address (0 if none) |
| Reader | 4 | beta_addr | uint32_t | Beta buffer address (0 if none) |
| Reader | 5 | scaler_value | uint32_t | Packed bf16 scaler (1/W) |
| Reader | 6 | eps_value | uint32_t | Packed bf16 epsilon |

**Runtime** (Writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | dst_addr | uint32_t | Output buffer DRAM address |
| Writer | 1 | num_tile_rows | uint32_t | Tile-rows for this core |
| Writer | 2 | start_stick_id | uint32_t | First output stick ID |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB c_2 is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- reduce/binary helpers handle auto-splitting
- [x] RM CBs count pages in sticks, tile CBs count in tiles
- [x] Gamma/beta tilize: reader duplicates single stick 32 times to fill one tile-row

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile | Tile iteration | `(1, 1, 64, 128)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 | Wt | tile-sized pages, RM data | All | Block |
| c_1 | Wt | TILE | All | Block |
| c_2 | 1 | TILE (bf16) | Row0 of each face | Program |
| c_3 | 1 | TILE (bf16) | All | Program |
| c_4 | Wt | TILE | All (gamma broadcast) | Program |
| c_5 | Wt | TILE | All (beta broadcast) | Program |
| c_24 | 1 | TILE | Col0 (reduce_row output) | Tile-row |
| c_25 | Wt | TILE | All | Phase |
| c_26 | 1 | TILE | Col0 (reduce_row output) | Tile-row |
| c_27 | 1 | TILE | Col0 (scalar result) | Tile-row |
| c_16 | Wt | TILE | All | Block |
| c_17 | Wt | tile-sized pages, RM data | All | Block |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| Sub mean (Pass 2,3) | SUB | All (c_1) | Col0 (c_24 mean) | COL |
| Square (Pass 2) | SQUARE | All (c_25) | All (c_25=self) | NONE |
| Add eps (Pass 3) | ADD+rsqrt | Col0 (c_26 var) | All (c_3 eps) | SCALAR |
| Mul inv_std (Pass 3) | MUL | All (c_25 centered) | Col0 (c_27 inv_std) | COL |
| Mul gamma (Pass 3) | MUL | All (c_16 normalized) | All (c_4 gamma) | NONE |
| Add beta (Pass 3) | ADD | All (c_25 scaled) | All (c_5 beta) | NONE |

Note on "Add eps": variance (c_26) has only Col0 valid, epsilon (c_3) is a scalar tile with all elements = eps. Using `add<BroadcastDim::SCALAR>` with A=c_26 and B=c_3. Since we only need the scalar result in one tile, BinaryInputBlockShape::single() suffices. The rsqrt is applied as a post_op callback.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | tilize + untilize (identity) | x | input shape | - |
| 2 | reduce_mean | reduce_row with 1/W scaler | mean(x, dim=-1) | tile-aligned reduced | [:,:,:,0:1] |
| 3 | subtract_mean | 2nd tilize + sub<COL> | x - mean(x) | input shape | - |
| 4 | variance | square + reduce_row | var(x, uncorrected) | tile-aligned reduced | [:,:,:,0:1] |
| 5 | normalize | add_eps+rsqrt + 3rd tilize + sub + mul | layer_norm(x) | input shape | - |
| 6 | affine | gamma/beta tilize + mul + add | layer_norm(x, gamma, beta) | input shape | - |

### Stage 1: data_pipeline
- **Scope**: reader (input sticks + scaler + eps), compute (tilize c_0->c_1, untilize c_1->c_17), writer (stick extraction)
- **Reference**: `x` (identity passthrough -- tilize then immediately untilize)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes c_0->c_1, then untilizes c_1->c_17. All other CBs unused. Reader still prepares scaler/eps CBs (no-op for compute).

### Stage 2: reduce_mean
- **Scope**: compute adds reduce<SUM, REDUCE_ROW> after tilize
- **Reference**: `x.mean(dim=-1, keepdim=True)` -- reduced shape, only Col0 valid
- **Output shape**: `list(shape[:-1]) + [32]` (tile-aligned W=32)
- **Compare slice**: `[:,:,:,0:1]`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from Stage 1**: Add reduce phase between tilize and untilize. Untilize input CB changes from c_1 to c_24.

### Stage 3: subtract_mean
- **Scope**: compute adds Pass 2 (tilize + sub<COL>)
- **Reference**: `x - x.mean(dim=-1, keepdim=True)` -- centered data, full shape
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from Stage 2**: After Pass 1 (tilize+reduce->mean), do Pass 2: tilize c_0->c_1 again, sub<COL>(c_1, c_24)->c_25, untilize c_25->c_17.

### Stage 4: variance
- **Scope**: compute adds square + reduce in Pass 2
- **Reference**: `((x - x.mean(dim=-1, keepdim=True))**2).mean(dim=-1, keepdim=True)`
- **Output shape**: `list(shape[:-1]) + [32]`
- **Compare slice**: `[:,:,:,0:1]`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from Stage 3**: After sub<COL>, add square(c_25)->c_1, reduce<SUM,REDUCE_ROW>(c_1)->c_26. Untilize c_26 instead of c_25.

### Stage 5: normalize
- **Scope**: compute adds Pass 3 (add_eps+rsqrt, tilize, sub, mul_inv_std)
- **Reference**: `torch.nn.functional.layer_norm(x, [shape[-1]], eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from Stage 4**: After variance, add_eps+rsqrt on c_26->c_27. Then Pass 3: tilize c_0->c_1, sub<COL>(c_1,c_24)->c_25, mul<COL>(c_25,c_27)->c_16, untilize c_16->c_17. Pop mean+var+inv_std.

### Stage 6: affine
- **Scope**: reader adds gamma/beta reading+duplication, compute adds gamma/beta tilize + mul + add
- **Reference**: `torch.nn.functional.layer_norm(x, [shape[-1]], weight=gamma.squeeze(), bias=beta.squeeze(), eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from Stage 5**: At program start, reader duplicates gamma/beta sticks 32x into c_0, compute tilizes c_0->c_4 and c_0->c_5. In Pass 3, after mul_inv_std to c_16, add mul(c_16, c_4)->c_25 and add(c_25, c_5)->c_16. Untilize c_16->c_17.

### Reader Kernel
Reads RM sticks from DRAM using TensorAccessor. Per tile-row, reads the same 32 sticks 3 times (3 passes for mean, variance, normalize). Each read fills c_0 with Wt tile-sized pages of RM data. At program start: fills c_2 with 1/W scaler via `prepare_reduce_scaler`, fills c_3 with epsilon via `prepare_reduce_scaler`. If gamma/beta provided: reads their single stick 32 times into c_0 (duplicating to fill one tile-row), pushes once each.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_2, c_1);`

#### Gamma/Beta Tilize (once at program start, if present)
```cpp
// Gamma
compute_kernel_lib::tilize<c_0, c_4, InitAndUninit>(Wt, 1);
// Beta
compute_kernel_lib::tilize<c_0, c_5, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(Wt, 1);
```
- c_4: Wt tiles, gamma persists for entire program (WaitUpfrontNoPop in binary ops)
- c_5: Wt tiles, beta persists for entire program (WaitUpfrontNoPop in binary ops)

**CB state after gamma/beta tilize:**
| CB | Tiles | State |
|----|-------|-------|
| c_4 | Wt | pushed, persistent for program |
| c_5 | Wt | pushed, persistent for program |

#### Main loop: for each tile-row (num_tile_rows iterations)

##### Phase 1: Tilize input (Pass 1)
```cpp
compute_kernel_lib::tilize<c_0, c_1, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(Wt, 1);
```

##### Phase 2: Reduce row -> mean
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitAndPopPerTile, INPUT_AND_OUTPUT>(
    c_1, c_2, c_24, ReduceInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, waited+popped per tile by reduce]
- Scaler: c_2 [1 tile, waited internally, not popped -- persistent]
- Out: c_24 [1 tile, pushed]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | 0 | freed (popped by reduce) |
| c_24 | 1 | pushed -- persists for Phases 4, 7 |

##### Phase 3: Tilize input (Pass 2)
```cpp
compute_kernel_lib::tilize<c_0, c_1, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(Wt, 1);
```

##### Phase 4: Subtract mean -> centered
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,   // A: c_1
    BinaryInputPolicy::WaitUpfrontNoPop,     // B: c_24 (mean persists)
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_1, c_24, c_25, BinaryInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, popped per tile]
- B: c_24 [1 tile, waited upfront, NOT popped -- persists for Phase 7]
- Out: c_25 [Wt tiles, pushed per tile]

##### Phase 5: Square centered
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_25, c_1, BinaryInputBlockShape::of(1, Wt));
```
- A: c_25 [Wt tiles, popped per tile]
- Out: c_1 [Wt tiles, pushed per tile]

##### Phase 6: Reduce row -> variance
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitAndPopPerTile, INPUT_AND_OUTPUT>(
    c_1, c_2, c_26, ReduceInputBlockShape::of(1, Wt));
```
- Out: c_26 [1 tile, pushed]

##### Phase 7: Add epsilon + rsqrt -> inv_std
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,    // A: c_26 (var, consumed)
    BinaryInputPolicy::NoWaitNoPop,          // B: c_3 (eps, persistent, already waited)
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_26, c_3, c_27, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) { rsqrt_tile_init(); rsqrt_tile(dst_idx); });
```
- A: c_26 [1 tile, waited+popped]
- B: c_3 [1 tile, NOT waited (NoWaitNoPop) -- already in CB from reader, persistent]
- Out: c_27 [1 tile, pushed]
- Post-op: rsqrt applied in-place in DEST before pack

**CRITICAL**: For Phase 7, c_3 (eps) was pushed by reader at program start. The first tile-row iteration must use WaitUpfrontNoPop for B to initially wait for the tile. Subsequent iterations: the tile is still there (never popped). To handle this cleanly, the compute kernel should `cb_wait_front(c_3, 1)` once before the main loop, then use NoWaitNoPop in all iterations.

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 | 1 | still held (not popped from Phase 4) |
| c_26 | 0 | freed (popped by add) |
| c_27 | 1 | pushed -- persists for Phase 9 |

##### Phase 8: Tilize input (Pass 3)
```cpp
compute_kernel_lib::tilize<c_0, c_1, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(Wt, 1);
```

##### Phase 9: Subtract mean (again)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,    // A: c_1
    BinaryInputPolicy::NoWaitNoPop,          // B: c_24 (mean, still in CB from Phase 4)
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_1, c_24, c_25, BinaryInputBlockShape::of(1, Wt));
```

##### Phase 10: Multiply by inv_std -> normalized
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,    // A: c_25 (centered)
    BinaryInputPolicy::WaitUpfrontNoPop,     // B: c_27 (inv_std, first wait)
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_25, c_27, c_16, BinaryInputBlockShape::of(1, Wt));
```

##### Phase 11: Multiply by gamma (conditional)
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::NONE,
        BinaryInputPolicy::WaitAndPopPerTile,    // A: c_16 (normalized)
        BinaryInputPolicy::NoWaitNoPop,          // B: c_4 (gamma, persistent)
        BinaryOutputPolicy::PerTile,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        c_16, c_4, c_25, BinaryInputBlockShape::of(1, Wt));
}
```
- If gamma: output goes to c_25. If no gamma: c_16 already has the result.

##### Phase 12: Add beta (conditional)
```cpp
if constexpr (has_beta) {
    uint32_t src_cb = has_gamma ? c_25 : c_16;
    compute_kernel_lib::add<BroadcastDim::NONE,
        BinaryInputPolicy::WaitAndPopPerTile,    // A: src_cb
        BinaryInputPolicy::NoWaitNoPop,          // B: c_5 (beta, persistent)
        BinaryOutputPolicy::PerTile,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        src_cb, c_5, c_16, BinaryInputBlockShape::of(1, Wt));
}
```
- Output always ends up in c_16 for untilize.

##### Phase 13: Untilize -> RM output
```cpp
compute_kernel_lib::untilize<Wt, c_16, c_17, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(1);
```

##### End of tile-row: Pop persistent per-tile-row CBs
```cpp
cb_pop_front(c_24, 1);  // mean
cb_pop_front(c_27, 1);  // inv_std
```

#### End of program: Pop persistent program CBs
```cpp
cb_pop_front(c_2, 1);   // scaler
cb_pop_front(c_3, 1);   // eps
if (has_gamma) cb_pop_front(c_4, Wt);
if (has_beta) cb_pop_front(c_5, Wt);
```

### Writer Kernel
Waits for Wt tile-sized pages in c_17 per tile-row. Extracts 32 RM sticks using `get_read_ptr(c_17)` + row offset arithmetic. Writes each stick to DRAM via `noc_async_write` using TensorAccessor for address resolution. Issues `noc_async_write_barrier()` per tile-row block. Pops c_17 after each block.

### Critical Notes
1. **Reader reads input 3 times per tile-row**: Same 32 sticks re-read from DRAM each pass. The NOC addresses for these sticks are resolved once and cached in a local array.
2. **Gamma/beta stick duplication**: Gamma and beta are single sticks (1 row of W elements). The reader reads the same stick 32 times to fill one tile-row worth of RM data for tilize. This happens once at program start.
3. **Epsilon CB first-wait**: The compute kernel must `cb_wait_front(c_3, 1)` once before the main loop. All subsequent uses of c_3 use NoWaitNoPop policy.
4. **Scaler CB**: The reduce helper waits for c_2 internally on each reduce call. Since c_2 is 1 tile and persistent, the reduce helper's wait is satisfied every time.
5. **Affine routing**: When both gamma and beta are present, normalized->c_16, mul_gamma(c_16,c_4)->c_25, add_beta(c_25,c_5)->c_16. When only gamma: mul(c_16,c_4)->c_25, then untilize c_25. When only beta: add(c_16,c_5)->c_25, then untilize c_25. The output CB for untilize input must be consistent -- use c_16 as the final pre-untilize CB, routing through c_25 for intermediate affine steps.

### Implementation Checklist
- [ ] Reader: RM stick reads (TensorAccessor), 3 passes/tile-row, scaler/eps fill, gamma/beta duplication
- [ ] Compute: 13 phases using helpers: tilize, reduce<SUM,REDUCE_ROW>, sub<COL>, square, add<SCALAR>+rsqrt, mul<COL>, mul<NONE>, add<NONE>, untilize
- [ ] Writer: stick extraction from untilized CB, TensorAccessor DRAM writes
- [ ] CB push/pop balance verified (persistent CBs popped at end)
