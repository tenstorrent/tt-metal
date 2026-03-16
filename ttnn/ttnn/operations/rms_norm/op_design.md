# Operation Design: rms_norm

## Overview
- **Operation Name**: rms_norm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), batch_norm (compute_core)

## Mathematical Definition
```
output[n,c,h,w] = input[n,c,h,w] * rsqrt(mean(input[n,c,h,:]^2) + epsilon) * gamma[w]
```
RMSNorm normalizes each row (last dimension) by its root mean square, then optionally scales by gamma. Reduction is along `dim=-1` with `keepdim=True`.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-6 | Numerical stability constant |
| gamma | Tensor | No | shape[-1] == input.shape[-1] | None | Per-element scale, always RM layout with shape (1,1,1,W) |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Layout | ROW_MAJOR_LAYOUT or TILE_LAYOUT |
| Memory | INTERLEAVED (DRAM) |
| Dtype | BFLOAT16 or FLOAT32 |
| Rank | >= 2 |
| H, W (TILE) | Divisible by 32 |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: Same as input
- **Layout**: Same as input (RM in -> RM out, TILE in -> TILE out)
- **Memory**: INTERLEAVED (DRAM)

### Edge Cases
| Condition | Behavior |
|-----------|----------|
| rank < 2 | ValueError |
| gamma.shape[-1] != input.shape[-1] | ValueError |
| gamma is None | Skip gamma multiply phase |
| W not divisible by 32 (TILE) | ValueError |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize_analysis | input_stage | Add gamma reading, epsilon/scaler fill, two-pass re-read |
| Compute | batch_norm_analysis | compute_core | Replace subtract-mean with square+reduce_row, add rsqrt broadcast-multiply |
| Writer | untilize_analysis | output_stage | Conditional untilize output, tile-by-tile output for TILE layout |

### Work Distribution
- **Work unit**: 1 tile-row (Wt tiles along width)
- **Grid**: 1x1 (single core)
- **Work per core**: All tile-rows: `NC * Ht` rows, each with `Wt` tiles
- **Remainder**: N/A (single core)

### Data Flow
Two-pass per tile-row: Pass 1 reads all Wt tiles to compute `mean(x^2)`. Pass 2 re-reads the same tiles to multiply by `rsqrt(mean + eps)` and optionally by gamma. Reader pushes tiles to cb_in; compute squares, reduces, then normalizes; writer drains output.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in_rm | RM sticks for tilize | Reader | Compute | Wt | Per tile-row (RM input only) |
| c_1 | cb_in | Tilized / tile input | Compute(tilize) or Reader | Compute | Wt | Per tile-row |
| c_2 | cb_x_sq | x^2 intermediate | Compute | Compute | 2 | Per-tile streaming |
| c_3 | cb_gamma_rm | Gamma RM sticks for tilize | Reader | Compute | Wt | Per tile-row (gamma only) |
| c_4 | cb_gamma | Tilized gamma | Compute(tilize) or N/A | Compute | 2 | Per-tile streaming |
| c_8 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | Program |
| c_9 | cb_eps | Epsilon tile | Reader | Compute | 1 | Program |
| c_16 | cb_out | Output tiled data | Compute | Compute(untilize) or Writer | Wt (RM) / 2 (TILE) | Per tile-row |
| c_17 | cb_out_rm | Untilized output sticks | Compute(untilize) | Writer | Wt | Per tile-row (RM output only) |
| c_24 | cb_reduce_out | mean(x^2) accumulator | Compute | Compute | 2 | Per tile-row |
| c_25 | cb_rms_inv | rsqrt(mean+eps) | Compute | Compute | 2 | Per tile-row |
| c_26 | cb_norm | Normalized output (pre-gamma) | Compute | Compute | 2 | Per-tile (gamma only) |

**Notes:**
- CBs c_0, c_3, c_17 allocated only when input/output is RM layout.
- c_26 allocated only when gamma is present; otherwise mul output goes directly to c_16.
- All tile-layout CBs use tile_size page size. RM CBs use tile_size page size (32 sticks packed as tile-equivalent).
- cb_scaler data format: always bfloat16 (reduce LLK requirement). cb_eps uses input data format.

### Kernel Arguments

**Compile-time (Reader):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_rm_input | uint32_t | 1 if input is RM, 0 if TILE |
| 1 | has_gamma | uint32_t | 1 if gamma present |
| 2 | stick_size | uint32_t | Full input stick width in bytes (RM only) |
| 3 | gamma_stick_size | uint32_t | Gamma stick width in bytes (gamma only) |
| 4+ | TensorAccessorArgs (input) | | |
| ... | TensorAccessorArgs (gamma) | | If gamma present |

**Runtime (Reader):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input tensor base address |
| 1 | gamma_addr | uint32_t | Gamma tensor base address (0 if absent) |
| 2 | num_rows | uint32_t | Total tile-rows (NC * Ht) |
| 3 | Wt | uint32_t | Tiles per row |
| 4 | num_sticks | uint32_t | Total sticks in input (RM only) |
| 5 | num_tiles | uint32_t | Total tiles in input (TILE only) |
| 6 | packed_scaler | uint32_t | Reduce scaler value (1/W as packed bf16/f32) |
| 7 | packed_eps | uint32_t | Epsilon value packed |

**Compile-time (Compute):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_rm_input | uint32_t | 1 if RM layout |
| 1 | has_gamma | uint32_t | 1 if gamma |
| 2 | Wt | uint32_t | Width in tiles |
| 3 | Ht | uint32_t | Height in tiles |
| 4 | NC | uint32_t | Product of batch dimensions |

**Runtime (Compute):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows | uint32_t | Total tile-rows to process |

**Compile-time (Writer):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | is_rm_output | uint32_t | 1 if output is RM |
| 1 | output_stick_size | uint32_t | Output stick width in bytes (RM only) |
| 2+ | TensorAccessorArgs (output) | | |

**Runtime (Writer):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output tensor base address |
| 1 | num_rows | uint32_t | Total tile-rows |
| 2 | Wt | uint32_t | Tiles per row |
| 3 | num_tiles | uint32_t | Total output tiles (TILE) or sticks (RM) |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB (c_8) is bfloat16
- [x] DEST register holds max 8 tiles (bf16) / 4 tiles (f32)
- [x] RM CBs count pages in tiles (tile-equivalent sizing for tilize compatibility)
- [x] For fp32: set fp32_dest_acc_en=true, UnpackToDestMode::UnpackToDestFp32

### Test Criteria
- Output shape matches input shape exactly
- Numerical accuracy vs PyTorch `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)` with gamma scaling

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
| c_0 (cb_in_rm) | Wt | RM sticks as tile pages | All | Per tile-row |
| c_1 (cb_in) | Wt | TILE | All | Per tile-row |
| c_2 (cb_x_sq) | 2 | TILE | All | Per-tile |
| c_3 (cb_gamma_rm) | Wt | RM sticks as tile pages | All | Per tile-row |
| c_4 (cb_gamma) | 2 | TILE | All | Per-tile |
| c_8 (cb_scaler) | 1 | TILE (bf16) | Row0 | Program |
| c_9 (cb_eps) | 1 | TILE | All (broadcast-filled) | Program |
| c_16 (cb_out) | Wt(RM)/2(TILE) | TILE | All | Per tile-row |
| c_17 (cb_out_rm) | Wt | RM sticks as tile pages | All | Per tile-row |
| c_24 (cb_reduce_out) | 2 | TILE | Col0 (REDUCE_ROW output) | Per tile-row |
| c_25 (cb_rms_inv) | 2 | TILE | Col0 (scalar broadcast source) | Per tile-row |
| c_26 (cb_norm) | 2 | TILE | All | Per-tile (gamma only) |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| square | MUL | All (cb_in) | All (cb_in, same) | NONE (self-mul) |
| add_eps | ADD | Col0 (cb_reduce_out) | All (cb_eps, broadcast-filled) | SCALAR |
| normalize | MUL | All (cb_in) | Col0 (cb_rms_inv) | COL |
| gamma_mul | MUL | All (cb_norm) | All (cb_gamma) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | Reader+writer, tilize/untilize, identity passthrough | input (identity) | Same as input | N/A |
| 2 | square_reduce | Square + reduce_row in compute | mean(x^2, dim=-1) | Tile-aligned reduced | `[:,:,:,0:1]` |
| 3 | rms_normalize | Add eps+rsqrt, re-read input, broadcast multiply | x/rms(x) | Same as input | N/A |
| 4 | gamma_scale | Gamma tilize + multiply | Full RMSNorm with gamma | Same as input | N/A |

### Stage 1: data_pipeline
- **Scope**: reader kernel, writer kernel, compute kernel (tilize + untilize phases only)
- **Reference**: `input` (identity)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **Compute**: If RM input: `tilize<Wt>(Ht_per_row)` then `untilize<Wt>(Ht_per_row)`. If TILE: identity passthrough (copy tiles from cb_in to cb_out via pack_tile/copy_tile or equivalent).
- **CB bypass**: No square/reduce/normalize phases. Compute reads from cb_in, writes directly to cb_out.

### Stage 2: square_reduce
- **Scope**: compute kernel adds square + reduce phases
- **Reference**: `(input ** 2).mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Output shape**: Tile-aligned: `list(shape[:-1]) + [32]`
- **Compare slice**: `[:,:,:,0:1]`
- **Delta**: Reader now fills cb_scaler (1/W). Compute adds square + reduce_row per tile-row. Writer outputs 1 tile per row instead of Wt.

### Stage 3: rms_normalize
- **Scope**: compute adds add_eps+rsqrt phase, re-read input, broadcast multiply
- **Reference**: `input / torch.sqrt((input ** 2).mean(dim=-1, keepdim=True) + 1e-6)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta**: Reader re-reads input for pass 2. Compute adds: add(reduce_out, eps) with rsqrt post-op, then mul<COL>(input, rms_inv). Writer outputs full shape again.

### Stage 4: gamma_scale
- **Scope**: reader adds gamma reading+tilize, compute adds gamma multiply
- **Reference**: `(input / torch.sqrt((input ** 2).mean(dim=-1, keepdim=True) + 1e-6)) * gamma`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta**: Reader pushes gamma tiles. Compute adds mul<ROW>(norm, gamma). cb_norm and cb_gamma CBs activated.

### Reader Kernel
Two-pass reader per tile-row. Pass 1: push Wt input tiles to cb_in (RM: read 32 sticks then tilize in compute; TILE: read tiles directly). Pass 2: re-push same Wt tiles. At startup: fill cb_scaler with 1/W value using `prepare_reduce_scaler`, fill cb_eps using broadcast fill. For gamma: push Wt gamma tiles per row (re-read each row) via same tilize-from-RM pattern.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in, cb_scaler, cb_out)` (three-arg form for srcA=cb_in, srcB=cb_scaler, ocb=cb_out)

#### Phase 1: Tilize (RM input only)
```cpp
compute_kernel_lib::tilize<Wt, c_0, c_1,
    tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode::WaitBlock>(1);
```
- Input: cb_in_rm [Wt pages, pushed by reader]
- Output: cb_in [Wt tiles, tilized]
- CB state: cb_in_rm freed, cb_in has Wt tiles

#### Phase 2: Square
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_1, c_2, BinaryInputBlockShape::of(1, Wt));
```
- A: cb_in [Wt tiles, FRESHLY PUSHED by tilize or reader, popped per tile]
- Out: cb_x_sq [Wt tiles pushed one at a time, consumed by reduce]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 (cb_in) | 0 | all popped by square |
| c_2 (cb_x_sq) | Wt | all pushed, awaiting reduce |

#### Phase 3: Reduce Row
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile>(
    c_2, c_8, c_24,
    ReduceInputBlockShape::row(Wt, 1));
```
- Input: cb_x_sq [Wt tiles, waited and popped per tile]
- Scaler: cb_scaler [1 tile, persistent, contains 1/W]
- Output: cb_reduce_out [1 tile, mean(x^2)]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_2 (cb_x_sq) | 0 | all consumed |
| c_24 (cb_reduce_out) | 1 | freshly pushed |

#### Phase 4: Add Epsilon + Rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR>(
    c_24, c_9, c_25,
    BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: cb_reduce_out [1 tile, waited and popped]
- B: cb_eps [1 tile, persistent -- use WaitUpfrontNoPop for B across all rows]
- Out: cb_rms_inv [1 tile, rsqrt(mean + eps)]

**Note**: cb_eps uses `WaitUpfrontNoPop` for B so it persists across all tile-rows. Manually pop at kernel end.

#### Phase 5: Re-tilize (RM input only, pass 2)
Same as Phase 1. Reader re-pushes sticks, compute re-tilizes to cb_in.

#### Phase 6: Normalize (broadcast multiply)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_1, c_25, cb_norm_or_out,
    BinaryInputBlockShape::of(1, Wt));
```
- A: cb_in [Wt tiles, popped per tile]
- B: cb_rms_inv [1 tile, waited upfront, NOT popped -- persists for all Wt tiles]
- Out: cb_norm (if gamma) or cb_out (if no gamma) [Wt tiles, pushed per tile]

After loop: `cb_pop_front(c_25, 1)` to release cb_rms_inv.

#### Phase 7: Gamma Tilize + Multiply (optional)
If gamma present, reader pushes gamma sticks to cb_gamma_rm, compute tilizes:
```cpp
compute_kernel_lib::tilize<Wt, c_3, c_4>(1);
```
Then element-wise multiply:
```cpp
compute_kernel_lib::mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_26, c_4, c_16,
    BinaryInputBlockShape::of(1, Wt));
```
- A: cb_norm [Wt tiles, streaming]
- B: cb_gamma [Wt tiles, streaming]
- Out: cb_out [Wt tiles]

#### Phase 8: Untilize (RM output only)
```cpp
compute_kernel_lib::untilize<Wt, c_16, c_17,
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::WaitBlock>(1);
```
- Input: cb_out [Wt tiles, accumulated from phase 6/7]
- Output: cb_out_rm [Wt pages of RM sticks]

### Writer Kernel
**TILE output**: Wait for tiles from cb_out one at a time, write to DRAM using TensorAccessor (tile-page addressing).

**RM output**: Wait for Wt pages from cb_out_rm (one tile-row of sticks). Write 32 sticks per tile-row block. Each stick is `W * elem_size` bytes. Use TensorAccessor with stick-based page size. Pre-compute base NoC addresses for 32 sticks, write all sticks in the block, barrier, pop. Pattern from untilize reference writer.

### Critical Notes
1. **Two-pass re-read**: Input tiles are read from DRAM twice per tile-row. Reader must track stick/tile IDs to re-read the same data. Use `start_stick_id` saved before pass 1, restored for pass 2.
2. **cb_scaler format**: Always bfloat16 regardless of input dtype. The reduce LLK requires bfloat16 scaler. Use `prepare_reduce_scaler<c_8>(1.0f / W)` which auto-detects format from CB.
3. **cb_eps persistence**: eps tile is waited once at compute start with WaitUpfrontNoPop on B in the add phase. Pop once at kernel end.
4. **Untilize requires Wt tiles accumulated**: For RM output, cb_out must have Wt pages. The normalize phase pushes tiles one by one; they accumulate. Untilize waits for all Wt before processing.
5. **Gamma re-read per row**: Gamma is [1,1,1,W]. For each tile-row, reader re-reads gamma from DRAM and pushes sticks. Compute tilizes and multiplies. This trades bandwidth for L1 (avoids persisting Wt gamma tiles).
6. **FP32 support**: When input is float32, set `fp32_dest_acc_en = true` and configure all compute CBs with `UnpackToDestMode::UnpackToDestFp32`. DEST limit drops to 4 tiles.

### Implementation Checklist
- [ ] Reader: TensorAccessor for input (sticks or tiles), gamma (sticks), eps/scaler fill
- [ ] Compute: 8 phases -- tilize, square, reduce<SUM,REDUCE_ROW>, add+rsqrt, re-tilize, mul<COL>, gamma tilize+mul, untilize
- [ ] Writer: TensorAccessor for output (sticks or tiles), conditional RM/TILE paths
- [ ] CB push/pop balance verified per tile-row iteration
