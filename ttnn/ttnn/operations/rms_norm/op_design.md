# Operation Design: rms_norm

## Overview
- **Operation Name**: rms_norm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), moreh_norm_w (compute_core)

## Mathematical Definition
```
output[n,c,h,w] = input[n,c,h,w] / sqrt(mean(input[n,c,h,:]^2) + epsilon) * gamma[w]
```
RMS (Root Mean Square) normalization along the last dimension. Gamma is optional; when absent, the `* gamma` term is omitted.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | ttnn.Tensor | Yes | rank >= 2, TILE: H,W % 32 == 0 | - | Input tensor |
| gamma | ttnn.Tensor | No | shape (1,1,1,W), ROW_MAJOR | None | Per-channel scale |
| epsilon | float | No | > 0 | 1e-6 | Numerical stability constant |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Layout | ROW_MAJOR_LAYOUT or TILE_LAYOUT |
| Memory | INTERLEAVED, DRAM |
| Dtype | BFLOAT16 or FLOAT32 |
| Rank | >= 2 |
| TILE_LAYOUT | H, W divisible by 32 |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: Same as input
- **Layout**: Same as input (RM in -> RM out, TILE in -> TILE out)
- **Memory**: INTERLEAVED, DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| rank < 2 | Python-side ValueError |
| gamma.shape[-1] != input.shape[-1] | Python-side ValueError |
| W == 32 (single tile width) | Wt=1, reduce operates on single tile |
| FLOAT32 input | fp32_dest_acc_en=true, intermediate CBs use Float32 |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader (RM) | tilize analysis | input_stage | Double-push per row; generate scaler/eps/gamma tiles |
| Reader (TILE) | moreh_norm_w analysis | input_stage | Tile-by-tile read; double-push per row |
| Compute | moreh_norm_w + helpers | compute_core | Replace manual accumulate with helper chain: square, reduce, add+rsqrt, mul_col, mul_row |
| Writer (RM) | untilize analysis | output_stage | Write RM sticks from untilized CB |
| Writer (TILE) | moreh_norm_w analysis | output_stage | Write tiles from output CB |

### Work Distribution
- **Work unit**: tile-row (32 rows x full W, producing Wt tiles)
- **Total units**: `(N * C * Ht)` where `Ht = H / 32`
- **Grid**: 1D linearized, up to `device.compute_with_storage_grid_size()`
- **Work per core**: `ceil(total_units / num_cores)`, cliff core gets remainder
- **Remainder**: Last core handles `total_units % units_per_core` rows

### Data Flow

Reader pushes each tile-row **twice** to `cb_in`: first for square+reduce, second for normalize. Persistent CBs (scaler, eps, gamma) are filled once at program start. If RM input, compute tilizes from `cb_in` to `cb_x` before each pass. Output path reverses: if RM, untilize before writer.

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in | Input staging (RM sticks or tiles) | Reader | Compute (tilize or square) | Wt | Block |
| c_1 | cb_scaler | Reduce scaler tile (1/W for SUM = mean) | Reader | Compute (reduce) | 1 | Program |
| c_2 | cb_eps | Epsilon scalar tile | Reader | Compute (add) | 1 | Program |
| c_3 | cb_gamma_rm | Gamma RM sticks (tilize staging) | Reader | Compute (tilize) | Wt | Once (if gamma) |
| c_4 | cb_gamma | Gamma tilized tiles | Compute (tilize) | Compute (mul_row) | Wt | Program (if gamma) |
| c_16 | cb_out | Final output (RM sticks or tiles) | Compute (untilize or mul) | Writer | Wt | Block |
| c_24 | cb_x | Tilized input (RM path only) | Compute (tilize) | Compute (square, mul_col) | Wt | Block |
| c_25 | cb_xsq | x^2 intermediate | Compute (square) | Compute (reduce) | 1 | Streaming |
| c_26 | cb_rms | Reduce output: mean(x^2) | Compute (reduce) | Compute (add+rsqrt) | 1 | Block |
| c_27 | cb_rsqrt | rsqrt(mean + eps) | Compute (add+rsqrt) | Compute (mul_col) | 1 | Block |
| c_28 | cb_normed | x * rsqrt (pre-gamma) | Compute (mul_col) | Compute (mul_row) | 1 | Streaming (if gamma) |

**CB page size**: All CBs use `tile_size(data_format)`. For RM input, cb_in's Wt tile-sized pages hold 32 sticks (same byte count).

**CB data format**: cb_in, cb_out, cb_gamma_rm, cb_gamma use input data format. cb_xsq, cb_rms, cb_rsqrt, cb_normed use intermed_format (Float32 if fp32_dest_acc_en, else same as input). cb_scaler and cb_eps always use bfloat16 (reduce scaler requirement).

**Layout-conditional CBs**: cb_x (c_24) and cb_normed (c_28) are only allocated when IS_INPUT_RM or HAS_GAMMA respectively. For TILE input, compute reads directly from cb_in (c_0) instead of cb_x. The compile-time routing:
- `compute_input = IS_INPUT_RM ? c_24 : c_0`
- When no gamma and TILE: mul_col writes directly to c_16
- When no gamma and RM: mul_col writes to c_24 (reuse), untilize(c_24, c_16)
- When gamma and TILE: mul_col writes to c_28, mul_row writes to c_16
- When gamma and RM: mul_col writes to c_28, mul_row writes to c_24 (reuse), untilize(c_24, c_16)

### Kernel Arguments

**Compile-time** (defines):
| Kernel | Define | Type | Description |
|--------|--------|------|-------------|
| All | IS_INPUT_RM | bool | Input is ROW_MAJOR_LAYOUT (controls tilize/untilize) |
| All | HAS_GAMMA | bool | Gamma tensor is provided |
| Compute | Wt | uint32_t | Tiles along W dimension |
| Compute | fp32_dest_acc_en | bool | Enable FP32 DEST accumulation |

**Compile-time** (per kernel):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | RM stick size in bytes (W * elem_size) |
| Reader | 1+ | TensorAccessorArgs(input) | varies | Input buffer addressing |
| Writer | 0 | stick_size | uint32_t | Output stick size in bytes |
| Writer | 1+ | TensorAccessorArgs(output) | varies | Output buffer addressing |

**Runtime** (per kernel):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer DRAM base address |
| Reader | 1 | num_rows | uint32_t | Tile-rows for this core |
| Reader | 2 | start_row_id | uint32_t | First tile-row index |
| Reader | 3 | Wt | uint32_t | Tiles along W |
| Reader | 4 | gamma_addr | uint32_t | Gamma buffer address (0 if no gamma) |
| Compute | 0 | num_rows | uint32_t | Tile-rows for this core |
| Compute | 1 | Wt | uint32_t | Tiles along W |
| Compute | 2 | origin_w | uint32_t | Original W (for scaler = 1/W) |
| Writer | 0 | dst_addr | uint32_t | Output buffer DRAM base address |
| Writer | 1 | num_rows | uint32_t | Tile-rows for this core |
| Writer | 2 | start_row_id | uint32_t | First tile-row index (for output page ID) |
| Writer | 3 | Wt | uint32_t | Tiles along W |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (1 for streaming CBs, Wt for block CBs)
- [x] Reduce scaler CB (c_1) is bfloat16
- [x] DEST register holds max 8 tiles (bf16) / 4 tiles (f32) -- helpers handle sub-blocking
- [x] RM CBs count pages in tiles (tile-sized pages for tilize compatibility)
- [x] Epsilon CB (c_2) is bfloat16 (scalar for reduce pattern)

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs PyTorch reference

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile | Tile iteration | `(1, 1, 64, 128)` |
| Non-square | W != H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (cb_in) | Wt | RM sticks or tiles | All | Block |
| c_1 (cb_scaler) | 1 | bfloat16 tile | Row0 (reduce scaler format) | Program |
| c_2 (cb_eps) | 1 | bfloat16 tile | [0,0] scalar | Program |
| c_4 (cb_gamma) | Wt | tiles | Row0 (gamma is 1-row) | Program (if gamma) |
| c_16 (cb_out) | Wt | RM sticks or tiles | All | Block |
| c_24 (cb_x) | Wt | tiles | All | Block (RM path only) |
| c_25 (cb_xsq) | 1 | intermed tiles | All | Streaming |
| c_26 (cb_rms) | 1 | intermed tiles | Col0 (REDUCE_ROW output) | Block |
| c_27 (cb_rsqrt) | 1 | intermed tiles | Col0 | Block |
| c_28 (cb_normed) | 1 | intermed tiles | All | Streaming (if gamma) |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| add_eps | ADD | Col0 (reduce out) | [0,0] (eps scalar) | SCALAR |
| mul_col | MUL | All (tilized x) | Col0 (rsqrt) | COL |
| mul_row | MUL | All (normed x) | Row0 (gamma) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | Reader + writer + tilize/untilize | Identity (x) | Same as input | None |
| 2 | square_reduce_mean | Square + reduce SUM (scaler=1/W) | mean(x^2, dim=-1) | [.., H, 32] | [:,:,:,0:1] |
| 3 | rms_norm_no_gamma | add_eps + rsqrt + mul_col + 2nd tilize | x / sqrt(mean(x^2)+eps) | Same as input | None |
| 4 | rms_norm_with_gamma | mul_row with gamma | Full RMS norm with gamma | Same as input | None |

### Stage 1: data_pipeline
- **Scope**: reader kernel, writer kernel, compute kernel (tilize + copy_tiles + untilize for RM; copy_tiles for TILE)
- **Reference**: `input_tensor`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **How it works**: Reader pushes Wt pages to cb_in (one push per row, not double). If RM: tilize(cb_in, cb_x). copy_tiles(cb_x, cb_out_tile) where cb_out_tile is c_24 for RM or c_16 for TILE. If RM: untilize(cb_out_tile, cb_out). Writer drains cb_out.

### Stage 2: square_reduce_mean
- **Scope**: compute adds square + reduce phases; reader adds scaler generation
- **Reference**: `torch.mean(input_tensor ** 2, dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from Stage 1**: Reader now generates reduce scaler (1/W) in cb_scaler. Compute replaces copy_tiles with: square(cb_x, cb_xsq) + reduce<SUM, REDUCE_ROW>(cb_xsq, cb_scaler, cb_rms). Reader pushes only once per row (no double-push). Output is reduced shape. If RM: untilize(cb_rms, cb_out) with Wt=1 (single output tile per row). If TILE: writer reads 1 tile per row from cb_rms directly.
- **Output shape**: `list(shape[:-1]) + [32]` (W -> 32 for tile alignment)
- **Compare slice**: `[:,:,:,0:1]` (only col 0 valid from REDUCE_ROW)

### Stage 3: rms_norm_no_gamma
- **Scope**: compute adds add_eps + rsqrt + mul_col; reader adds epsilon generation + double-push
- **Reference**: `input_tensor / torch.sqrt(torch.mean(input_tensor ** 2, dim=-1, keepdim=True) + 1e-6)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from Stage 2**: Reader generates eps tile in cb_eps. Reader double-pushes each row. Compute adds: add<SCALAR>(cb_rms, cb_eps, cb_rsqrt) with rsqrt post-op, then second tilize pass (if RM), then mul<COL>(cb_x, cb_rsqrt, cb_out_tile). Output shape matches input.

### Stage 4: rms_norm_with_gamma
- **Scope**: compute adds mul_row for gamma; reader adds gamma tilize
- **Reference**: `(input_tensor / torch.sqrt(torch.mean(input_tensor ** 2, dim=-1, keepdim=True) + 1e-6)) * gamma`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from Stage 3**: Reader reads gamma RM data and tilizes it (once at program start). Compute adds mul<ROW>(cb_normed, cb_gamma, cb_out_tile).

### Reader Kernel

**RM path**: Uses TensorAccessor for stick-based reads. Per row: reserve Wt tile-pages in cb_in, DMA 32 sticks, push. Repeat for second pass (normalize). Gamma: read 1 gamma stick, zero-fill 31 padding sticks in cb_gamma_rm, push Wt pages.

**TILE path**: Uses TensorAccessor for tile reads. Per row: read Wt tiles one at a time into cb_in (or cb_x for simplified routing). Repeat for second pass. Gamma: same RM pattern regardless of input layout.

**Persistent tiles** (generated once at program start):
- cb_scaler (c_1): `prepare_reduce_scaler<c_1>(1.0f / origin_w)` from reduce_helpers_dataflow.hpp
- cb_eps (c_2): `prepare_reduce_scaler<c_2>(epsilon)` (reuse the scaler fill utility for epsilon)

### Compute Kernel

**Startup**: `binary_op_init_common(cb_x, cb_x, cb_out)` (covers most helpers; individual helpers reconfigure as needed)

**Persistent CB setup** (once):
```cpp
cb_wait_front(cb_scaler, 1);  // scaler persists
cb_wait_front(cb_eps, 1);     // eps persists
if constexpr (HAS_GAMMA) {
    // tilize gamma: cb_gamma_rm -> cb_gamma
    tilize<cb_gamma_rm, cb_gamma>(Wt, 1);
    cb_wait_front(cb_gamma, Wt);  // gamma persists
}
```

**Per-row loop**:

#### Phase 1 (RM only): Tilize input pass 1
```cpp
compute_kernel_lib::tilize<cb_in, cb_x>(Wt, 1);
```
- In: cb_in [Wt pages, pushed by reader]
- Out: cb_x [Wt tiles, pushed by tilize]

#### Phase 2: Square
```cpp
compute_kernel_lib::square(cb_x, cb_xsq, BinaryInputBlockShape::row(Wt));
```
- A: cb_x [Wt tiles, streaming WaitAndPopPerTile -- consumed]
- Out: cb_xsq [1 tile at a time, streaming]

#### Phase 3: Reduce mean (SUM with 1/W scaler)
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
    cb_xsq, cb_scaler, cb_rms,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- In: cb_xsq [1 tile at a time, WaitAndPopPerTile]
- Scaler: cb_scaler [1 tile, already waited, persists]
- Out: cb_rms [1 tile, pushed]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| cb_x | 0 | freed (consumed by square) |
| cb_xsq | 0 | freed (consumed by reduce) |
| cb_rms | 1 | freshly pushed |
| cb_scaler | 1 | persistent (waited, not popped) |

#### Phase 4: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<
    BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,   // A = cb_rms (consumed)
    BinaryInputPolicy::NoWaitNoPop>(        // B = cb_eps (persistent)
    cb_rms, cb_eps, cb_rsqrt, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: cb_rms [1 tile, consumed]
- B: cb_eps [1 tile, persistent via NoWaitNoPop -- pre-waited at program start]
- Out: cb_rsqrt [1 tile, pushed] -- contains rsqrt(mean(x^2) + eps) in all elements (Col0 is the meaningful part)

#### Phase 5 (RM only): Tilize input pass 2
```cpp
compute_kernel_lib::tilize<cb_in, cb_x>(Wt, 1);
```
Reader has pushed the same row's sticks again. Produces Wt tiles in cb_x.

#### Phase 6: Multiply x * rsqrt (COL broadcast)
```cpp
compute_kernel_lib::mul<
    BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,   // A = cb_x (consumed)
    BinaryInputPolicy::WaitUpfrontPopAtEnd>( // B = cb_rsqrt (1 tile, consumed at end)
    cb_x, cb_rsqrt, cb_normed_out, BinaryInputBlockShape::row(Wt));
```
- A: cb_x [Wt tiles, streaming, consumed]
- B: cb_rsqrt [1 tile, waited upfront, popped at end of this row]
- Out: cb_normed_out [1 tile at a time, streaming]
- `cb_normed_out = HAS_GAMMA ? cb_normed : final_tile_cb`

#### Phase 7 (if gamma): Multiply by gamma (ROW broadcast)
```cpp
compute_kernel_lib::mul<
    BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,  // A = cb_normed (consumed)
    BinaryInputPolicy::NoWaitNoPop>(       // B = cb_gamma (persistent)
    cb_normed, cb_gamma, final_tile_cb, BinaryInputBlockShape::row(Wt));
```
- A: cb_normed [Wt tiles, streaming, consumed]
- B: cb_gamma [Wt tiles, persistent via NoWaitNoPop -- pre-waited at program start]
- Out: final_tile_cb [1 tile at a time, streaming]

#### Phase 8 (RM only): Untilize output
```cpp
compute_kernel_lib::untilize<Wt, final_tile_cb, cb_out>(1);
```
- In: final_tile_cb [Wt tiles, waited and consumed by untilize]
- Out: cb_out [Wt tile-pages of RM data, pushed]

**Cleanup** (once at program end):
```cpp
cb_pop_front(cb_scaler, 1);
cb_pop_front(cb_eps, 1);
if constexpr (HAS_GAMMA) { cb_pop_front(cb_gamma, Wt); }
```

### Writer Kernel

**RM path**: Per row: cb_wait_front(cb_out, Wt), extract 32 sticks, write each via TensorAccessor + noc_async_write, barrier, cb_pop_front. Pattern from untilize analysis.

**TILE path**: Per row: read Wt tiles from cb_out one at a time, write each via TensorAccessor + noc_async_write_tile. Pattern from moreh_norm_w writer.

### Critical Notes

1. **Double-push**: Reader must push each tile-row twice. First push feeds square+reduce. Second push feeds normalize (mul_col). The reader's per-row loop has two identical read passes.

2. **Gamma tilize padding**: Gamma has shape (1,1,1,W) = 1 stick. Reader must zero-fill 31 rows in cb_gamma_rm before pushing so tilize produces valid tiles. Use DMA from MEM_ZEROS_BASE to zero the CB, then write gamma stick at offset 0.

3. **Persistent CB lifecycle**: cb_scaler, cb_eps, cb_gamma are waited once at program start, never popped during the main loop, and popped once at program end. Helpers using NoWaitNoPop policy for these CBs rely on this pre-wait.

4. **Epsilon/scaler CB format**: Both cb_scaler (c_1) and cb_eps (c_2) must be bfloat16 format. The `prepare_reduce_scaler` function fills row 0 of faces in the correct format for reduce operations. For epsilon, this same function can be reused to create a scalar tile. The add helper will correctly broadcast the value.

5. **Untilize block_width_tiles**: The untilize helper requires `block_width_tiles` as a compile-time template parameter (= Wt). Since Wt is a runtime value in the general case, it must be passed as a compile-time arg to the compute kernel. This is standard practice.

6. **Output CB routing**: The destination CB for Phase 6 (mul_col) depends on whether gamma is present and whether output is RM. Use compile-time constexpr routing:
   - `final_tile_cb = IS_INPUT_RM ? cb_x : cb_out` (reuse cb_x as pre-untilize staging in RM path)
   - When HAS_GAMMA: Phase 6 outputs to cb_normed (c_28), Phase 7 outputs to `final_tile_cb`
   - When no gamma: Phase 6 outputs directly to `final_tile_cb`

### Implementation Checklist
- [ ] Reader: TensorAccessor for input, generate scaler/eps tiles, double-push per row, gamma tilize staging
- [ ] Compute: 8 phases using helpers: tilize, square, reduce<SUM, REDUCE_ROW>, add<SCALAR>+rsqrt, tilize (2nd), mul<COL>, mul<ROW>, untilize
- [ ] Writer: TensorAccessor for output, RM stick extraction or tile write
- [ ] CB push/pop balance verified: all persistent CBs popped at program end
