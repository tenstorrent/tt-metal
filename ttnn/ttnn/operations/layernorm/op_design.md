# Operation Design: layernorm

## Overview
- **Operation Name**: layernorm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), softmax (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[h] = sum(x[h, :]) / W
var[h] = sum((x[h, :] - mean[h])^2) / W
y[h, w] = (x[h, w] - mean[h]) / sqrt(var[h] + eps) * gamma[w] + beta[w]
```
Per-row normalization of a 2D [H, W] input. Gamma/beta are optional affine parameters.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| eps | float | No | > 0 | 1e-5 | Variance stabilizer |
| gamma | Tensor | No | 1D [W], bf16, RM | None | Scale parameter |
| beta | Tensor | No | 1D [W], bf16, RM | None | Shift parameter |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Rank | 2D [H, W] | "Input must be 2D" |
| Layout | ROW_MAJOR | "Input must be row-major" |
| Dtype | BFLOAT16 | "Only bfloat16 supported" |
| Memory | DRAM interleaved | "Input must be interleaved DRAM" |
| W alignment | Multiple of 32 (TILE_WIDTH) | "Width must be tile-aligned" |
| H alignment | Multiple of 32 (TILE_HEIGHT) | "Height must be tile-aligned" |

### Output Tensor Specification
- **Shape**: Same as input [H, W]
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: DRAM interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| gamma=None, beta=None | Skip affine transform phases |
| gamma only | Scale without shift |
| beta only | Shift without scale |
| W=32 (single tile width) | Wt=1, reduce is trivial |
| eps very small | rsqrt may overflow; user responsibility |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize analysis | input_stage | Add gamma/beta reads; batch 32 sticks per tile-row |
| Compute | softmax analysis + new | compute_core | Replace max/exp/recip with mean/var/rsqrt/affine |
| Writer | untilize analysis | output_stage | Write RM sticks from untilized output CB |

### Work Distribution
- **Work unit**: Tile-row (32 sticks x full W = Wt tiles)
- **Grid**: 1D linear, up to full device compute grid
- **Work per core**: `nblocks_per_core` tile-rows via `split_blocks_for_tilize`
- **Remainder**: Cliff core gets `nblocks % nblocks_per_core` tile-rows

### Data Flow
Reader reads 32 RM sticks into cb_in per tile-row. Compute tilizes, performs 7-phase normalization pipeline, untilizes. Writer writes 32 RM output sticks per tile-row.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in | RM input sticks | Reader | Compute(tilize) | Wt | Block |
| c_1 | cb_tilized | Tilized input | Compute | Compute | Wt | Row (persist for mean+centered) |
| c_2 | cb_reduce_scaler | Reduce scaler (1/W baked in) | Reader | Compute | 1 | Program |
| c_3 | cb_mean | Row mean (1 tile) | Compute | Compute | 1 | Row |
| c_4 | cb_centered | x - mean | Compute | Compute | Wt | Row (persist for var+normalize) |
| c_5 | cb_var | Row variance (1 tile) | Compute | Compute | 1 | Row |
| c_6 | cb_gamma | Gamma tiles (tilized by reader) | Reader | Compute | Wt | Program |
| c_7 | cb_beta | Beta tiles (tilized by reader) | Reader | Compute | Wt | Program |
| c_8 | cb_eps | Epsilon tile | Reader | Compute | 1 | Program |
| c_16 | cb_normalized | Normalized output (tiled) | Compute | Compute(untilize) | Wt | Block |
| c_17 | cb_out | RM output sticks | Compute(untilize) | Writer | Wt | Block |

### Kernel Arguments

**Compile-time** (Reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | W * 2 bytes |
| Reader | 1 | has_gamma | uint32_t | 0 or 1 |
| Reader | 2 | has_beta | uint32_t | 0 or 1 |
| Reader | 3+ | TensorAccessorArgs(input) | ... | Input accessor |
| Reader | N+ | TensorAccessorArgs(gamma) | ... | Gamma accessor (if present) |
| Reader | M+ | TensorAccessorArgs(beta) | ... | Beta accessor (if present) |

**Runtime** (Reader):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer address |
| Reader | 1 | num_sticks | uint32_t | Total sticks for this core |
| Reader | 2 | Wt | uint32_t | Width in tiles |
| Reader | 3 | start_stick_id | uint32_t | First stick index for this core |
| Reader | 4 | gamma_addr | uint32_t | Gamma buffer address (0 if none) |
| Reader | 5 | beta_addr | uint32_t | Beta buffer address (0 if none) |
| Reader | 6 | eps_value | uint32_t | Epsilon as reinterpreted uint32 |

**Compile-time** (Compute):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Compute | 0 | Wt | uint32_t | Width in tiles |
| Compute | 1 | nblocks_per_core | uint32_t | Tile-rows per core |
| Compute | 2 | has_gamma | uint32_t | 0 or 1 |
| Compute | 3 | has_beta | uint32_t | 0 or 1 |

**Compile-time** (Writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | stick_size | uint32_t | W * 2 bytes |
| Writer | 1+ | TensorAccessorArgs(output) | ... | Output accessor |

**Runtime** (Writer):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Writer | 0 | dst_addr | uint32_t | Output buffer address |
| Writer | 1 | num_sticks | uint32_t | Total sticks for this core |
| Writer | 2 | Wt | uint32_t | Width in tiles |
| Writer | 3 | start_stick_id | uint32_t | First output stick for this core |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB is bfloat16
- [x] DEST register holds max 8 tiles (bf16) / 4 tiles (f32)
- [x] RM CBs count pages in sticks, tile CBs count in tiles

### Test Criteria
- Output shape matches input shape [H, W]
- Numerical accuracy vs `torch.nn.functional.layer_norm(input, [W], weight=gamma, bias=beta, eps=eps)`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration | `(1, 1, 32, 128)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-row | Multiple tile-rows | `(1, 1, 64, 128)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (cb_in) | Wt | RM (tile-sized pages) | All | Block |
| c_1 (cb_tilized) | Wt | Tile | All | Row |
| c_2 (cb_reduce_scaler) | 1 | Tile (bf16) | Row0 | Program |
| c_3 (cb_mean) | 1 | Tile | Col0 | Row |
| c_4 (cb_centered) | Wt | Tile | All | Row |
| c_5 (cb_var) | 1 | Tile | Col0 | Row |
| c_6 (cb_gamma) | Wt | Tile | Row0 | Program |
| c_7 (cb_beta) | Wt | Tile | Row0 | Program |
| c_8 (cb_eps) | 1 | Tile | Scalar [0,0] | Program |
| c_16 (cb_normalized) | Wt | Tile | All | Block |
| c_17 (cb_out) | Wt | RM (tile-sized pages) | All | Block |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 3 (x - mean) | SUB | All (cb_tilized) | Col0 (cb_mean) | COL |
| 5 (centered^2) | SQUARE | All (cb_centered) | N/A | N/A |
| 7 ((x-mean) * rsqrt_var) | MUL | All (cb_centered) | Col0 (cb_var) | COL |
| 8 (norm * gamma) | MUL | All (cb_normalized) | Row0 (cb_gamma) | ROW |
| 9 (scaled + beta) | ADD | All (cb_normalized) | Row0 (cb_beta) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | tilize + untilize passthrough | input (identity) |
| 2 | subtract_mean | + reduce_sum for mean, subtract | x - mean(x) |
| 3 | normalize | + variance, rsqrt, full normalize | (x - mean) / sqrt(var + eps) |
| 4 | affine | + gamma * norm + beta | Full layernorm |

### Stage 1: data_pipeline
- **Scope**: reader, compute (tilize + untilize only), writer
- **Reference**: `input_tensor` (identity passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes cb_in -> cb_tilized, then copies cb_tilized -> cb_normalized (via cb_wait_front + untilize), untilizes cb_normalized -> cb_out. No normalization phases.

### Stage 2: subtract_mean
- **Scope**: compute adds reduce_sum + sub_bcast_cols
- **Reference**: `input_tensor - input_tensor.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Add Phase 2 (reduce_sum for mean) and Phase 3 (subtract mean broadcast)

### Stage 3: normalize
- **Scope**: compute adds square, reduce_sum for variance, add eps, rsqrt, multiply
- **Reference**: `torch.nn.functional.layer_norm(input_tensor, [input_tensor.shape[-1]], eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Add Phases 4-7 (square, reduce variance, add eps + rsqrt, multiply)

### Stage 4: affine
- **Scope**: reader adds gamma/beta reads; compute adds mul_bcast_rows + add_bcast_rows
- **Reference**: `torch.nn.functional.layer_norm(input_tensor, [input_tensor.shape[-1]], weight=gamma, bias=beta, eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Add gamma/beta reading in reader, Phases 8-9 in compute

### Reader Kernel
Reads 32 RM sticks per tile-row into cb_in (same pattern as tilize reader). On first tile-row, also reads gamma/beta as RM sticks into cb_gamma/cb_beta (Wt tiles each, persisted for program lifetime). Generates reduce scaler tile via `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, SUM, REDUCE_ROW, W>()`. Generates epsilon tile via `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps)`.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in, cb_reduce_scaler, cb_out)`

Per tile-row iteration (nblocks_per_core iterations):

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<cb_in, cb_tilized,
    tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);
```
- In: cb_in [Wt tiles, waited by helper, popped by helper]
- Out: cb_tilized [Wt tiles, pushed by helper]

#### Phase 2: Compute Mean (reduce sum * 1/W)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    cb_tilized, cb_reduce_scaler, cb_mean,
    ReduceInputBlockShape::row(Wt));
```
- In: cb_tilized [Wt tiles, WAITED but NOT POPPED -- persists for Phase 3]
- Scaler: cb_reduce_scaler [1 tile, baked-in 1/W, never popped]
- Out: cb_mean [1 tile, pushed]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| cb_tilized | Wt | waited, not popped -- persists for Phase 3 |
| cb_mean | 1 | freshly pushed |

#### Phase 3: Subtract Mean (x - mean)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_tilized, cb_mean, cb_centered,
    BinaryInputBlockShape::row(Wt));
```
- A: cb_tilized [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop]
- B: cb_mean [1 tile, WaitUpfrontPopAtEnd -- consumed]
- Out: cb_centered [Wt tiles, Bulk push]

Manual pop after Phase 3:
```cpp
cb_pop_front(cb_tilized, Wt);  // Release input tiles
```

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| cb_tilized | 0 | freed |
| cb_mean | 0 | freed (popped by PopAtEnd) |
| cb_centered | Wt | freshly pushed -- persists for Phases 4 and 7 |

#### Phase 4: Square Centered Values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_centered, cb_var, // cb_var reused as temp for squared output
    BinaryInputBlockShape::row(Wt));
```

Note: cb_var is used here as temporary storage for (x-mean)^2. We need a dedicated temp CB for this. Use c_5 as cb_squared_temp with Wt pages, then reduce into cb_var (1 page).

Correction: We need a separate CB for squared output since cb_var is 1-tile. Use cb_normalized (c_16, Wt pages) as temporary for squared values since it is not yet in use.

```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_centered, cb_normalized,  // c_16 as temp for (x-mean)^2
    BinaryInputBlockShape::row(Wt));
```
- A: cb_centered [Wt tiles, WAITED, NOT POPPED -- persists for Phase 7]
- Out: cb_normalized [Wt tiles as temp, Bulk push]

#### Phase 5: Compute Variance (reduce sum of squares * 1/W)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    cb_normalized, cb_reduce_scaler, cb_var,
    ReduceInputBlockShape::row(Wt));
```
- In: cb_normalized [Wt tiles temp squared, WAITED, NOT POPPED]
- Out: cb_var [1 tile, pushed]

Manual pop:
```cpp
cb_pop_front(cb_normalized, Wt);  // Release squared temp
```

#### Phase 6: Add Epsilon + Rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    true, NoAccumulation,
    decltype([](uint32_t dst_idx) { rsqrt_tile_init(); rsqrt_tile(dst_idx); })>(
    cb_var, cb_eps, cb_var,  // in-place: var -> rsqrt(var+eps)
    BinaryInputBlockShape::single(),
    {},
    {},
    [](uint32_t dst_idx) { rsqrt_tile_init(); rsqrt_tile(dst_idx); });
```

Note: In-place CB reuse (cb_var as both input and output) requires that the input is fully consumed before output is written. Since BinaryInputBlockShape::single() processes just 1 tile and PopAtEnd pops before Bulk push would write, this works. The output policy PerTile does reserve->pack->push per tile, and the input is popped at end. Since it's a single tile, the pop happens after pack, which means the input buffer space is freed after the value is in DST. This is safe.

cb_var now contains rsqrt(variance + eps) as a 1-tile COL-broadcast-ready value.

#### Phase 7: Normalize (centered * rsqrt_var)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_centered, cb_var, cb_normalized,
    BinaryInputBlockShape::row(Wt));
```
- A: cb_centered [Wt tiles, ALREADY WAITED from Phase 4, NoWaitPopAtEnd -- consumed]
- B: cb_var [1 tile, WaitUpfrontPopAtEnd -- consumed]
- Out: cb_normalized [Wt tiles, Bulk push]

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| cb_centered | 0 | freed |
| cb_var | 0 | freed |
| cb_normalized | Wt | freshly pushed |

#### Phase 8: Scale by Gamma (conditional)
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::ROW,
        BinaryInputPolicy::WaitUpfrontPopAtEnd,
        BinaryInputPolicy::NoWaitNoPop,
        BinaryOutputPolicy::Bulk,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        cb_normalized, cb_gamma, cb_normalized,  // in-place
        BinaryInputBlockShape::row(Wt));
}
```
- A: cb_normalized [Wt tiles, WaitUpfrontPopAtEnd -- consumed then rewritten]
- B: cb_gamma [Wt tiles, program-persistent, NoWaitNoPop -- never popped]

Note: In-place reuse requires cb_normalized to have 2*Wt capacity for safe double-buffering, OR the operation must fully read A before writing output. With WaitUpfrontPopAtEnd for A and Bulk for output, A is fully read first (upfront wait + indexed access), then popped, then output is bulk-pushed. This is safe with Wt capacity since all reads complete before any writes to the same CB.

#### Phase 9: Shift by Beta (conditional)
```cpp
if constexpr (has_beta) {
    compute_kernel_lib::add<BroadcastDim::ROW,
        BinaryInputPolicy::WaitUpfrontPopAtEnd,
        BinaryInputPolicy::NoWaitNoPop,
        BinaryOutputPolicy::Bulk,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        cb_normalized, cb_beta, cb_normalized,  // in-place
        BinaryInputBlockShape::row(Wt));
}
```
Same pattern as Phase 8.

#### Phase 10: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_normalized, cb_out,
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::NoWait>(1);
```
- In: cb_normalized [Wt tiles, ALREADY available from Phase 7/8/9, NoWait]
- Out: cb_out [Wt tiles RM, pushed by helper]

Note: After untilize, cb_normalized is internally popped by the helper.

### Writer Kernel
Waits for cb_out (Wt tiles per tile-row). For each block: `cb_wait_front(cb_out, Wt)`, then extracts 32 sticks from L1 read pointer, writes each stick via TensorAccessor + `noc_async_write`, calls `noc_async_write_barrier()`, then `cb_pop_front(cb_out, Wt)`. Same pattern as untilize writer.

### Critical Notes
1. **cb_tilized persistence**: Must survive through Phase 2 (reduce) and Phase 3 (subtract). WaitUpfrontNoPop in Phase 2 keeps tiles, manual pop after Phase 3.
2. **cb_centered persistence**: Must survive through Phase 4 (square) and Phase 7 (normalize multiply). WaitUpfrontNoPop in Phase 4, NoWaitPopAtEnd in Phase 7.
3. **cb_normalized reuse as temp**: Phase 4 writes squared values to cb_normalized (c_16), Phase 5 reduces them, then Phase 5 manually pops. Phase 7 writes final normalized output to same CB.
4. **In-place CB operations** (Phases 6, 8, 9): Safe because WaitUpfrontPopAtEnd reads all tiles first via indexed access, then pops, then Bulk output writes. The tiles are in DST registers between pop and push.
5. **Gamma/beta as RM 1D**: Reader must tilize gamma/beta. Read W elements as 1 stick, reader writes to cb_gamma/cb_beta. Compute tilizes these once at startup OR reader pre-tilizes them. Simplest: reader reads gamma as Wt tiles (each 32 elements wide) already in tile format by reading 32 identical rows. Alternative: store as RM and let compute handle ROW broadcast which reads Row0 only -- this works because ROW broadcast replicates the top row across all 32 rows of the tile. Reader fills gamma/beta CBs with Wt tile-sized pages where row 0 contains the data.
6. **Reduce scaler**: Uses `calculate_and_prepare_reduce_scaler<cb_reduce_scaler, SUM, REDUCE_ROW, W>()` which bakes in 1/W for AVG behavior. Since we want SUM and then multiply by 1/W ourselves: use SUM type with scaler=1.0, then the mean is obtained. Actually, using `AVG, REDUCE_ROW, W` directly computes mean in one step. Use `AVG` pool type.

### Implementation Checklist
- [ ] Reader: RM stick reads (32 per block), gamma/beta tile reads, scaler generation, eps generation
- [ ] Compute: 10 phases using helpers: tilize, reduce(AVG,REDUCE_ROW), sub(COL), square, reduce(SUM,REDUCE_ROW), add(SCALAR)+rsqrt, mul(COL), mul(ROW), add(ROW), untilize
- [ ] Writer: RM stick writes (32 per block) via TensorAccessor
- [ ] CB push/pop balance verified per tile-row iteration
