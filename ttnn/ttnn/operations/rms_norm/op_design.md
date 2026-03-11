# Operation Design: rms_norm

## Overview
- **Operation Name**: rms_norm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), reduce_w (compute_core)

## Mathematical Definition
```
output[n,c,h,w] = input[n,c,h,w] * rsqrt(mean(input[n,c,h,:]^2) + epsilon) * gamma[w]
```
RMS normalization along the last dimension. Gamma is optional; epsilon defaults to 1e-6.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | rank >= 2, bf16/f32 | - | Input tensor |
| gamma | Tensor | No | shape (1,1,1,W), RM layout | None | Scale parameter |
| epsilon | float | No | > 0 | 1e-6 | Stability constant |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Rank | >= 2 |
| Layout | ROW_MAJOR or TILE_LAYOUT |
| Memory | INTERLEAVED, DRAM |
| Dtype | BFLOAT16 or FLOAT32 |
| TILE_LAYOUT | H and W divisible by 32 |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: Same as input
- **Layout**: Same as input (RM in -> RM out, TILE in -> TILE out)
- **Memory**: INTERLEAVED, DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| rank < 2 | Python ValueError |
| gamma.shape[-1] != input.shape[-1] | Python ValueError |
| gamma=None | Skip gamma multiply phase |
| W = 32 (single tile width) | Wt=1, reduce over 1 tile |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader (RM) | tilize | input_stage | Add gamma stick reads, scaler/eps generation |
| Reader (TILE) | reduce_w | input_stage | Tile-by-tile reads into cb_tilized |
| Compute | reduce_w + new | compute_core | Multi-phase: tilize, square, reduce, add_eps_rsqrt, mul, gamma_mul, untilize |
| Writer (RM) | untilize | output_stage | Stick extraction from untilized CB |
| Writer (TILE) | reduce_w | output_stage | Tile-by-tile writes from cb_out |

### Work Distribution
- **Work unit**: tile-row (one row of Wt tiles = 32 sticks of width W)
- **Grid**: 1D linear from compute_with_storage_grid_size
- **Total units**: Ht_total = product(shape[:-1]) / 32
- **Work per core**: ceil(Ht_total / num_cores), with cliff core for remainder
- **Remainder**: Two core groups (group_1 gets ceil, group_2 gets floor)

### Data Flow

Per tile-row, the compute kernel executes a multi-phase pipeline. RM inputs are tilized in-kernel; RM outputs are untilized in-kernel. Gamma (always RM) is tilized once before the main loop. Input tiles persist in cb_tilized for reuse in both square and normalize-multiply phases.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input_rm | RM sticks for tilize (RM path only) | Reader | Compute (tilize) | Wt | Row |
| c_1 | cb_tilized | Tilized input (persistent per row) | Compute/Reader | Compute (square, mul) | Wt | Row (persistent) |
| c_2 | cb_scaler | Reduce scaler (1/W) | Reader | Compute (reduce) | 1 | Program |
| c_3 | cb_sq | Squared tiles | Compute (square) | Compute (reduce) | 2 | Streaming |
| c_4 | cb_rms | Reduce output: mean(x^2) | Compute (reduce) | Compute (add_eps) | 2 | Row |
| c_5 | cb_eps | Epsilon constant | Reader | Compute (add_eps) | 1 | Program |
| c_6 | cb_rms_inv | rsqrt(mean+eps) | Compute (add_eps_rsqrt) | Compute (mul) | 1 | Row |
| c_7 | cb_gamma | Gamma weights (tilized) | Compute (tilize) | Compute (gamma_mul) | Wt | Program |
| c_16 | cb_out | Output tiles | Compute (mul/gamma) | Writer/Compute (untilize) | Wt | Row |
| c_17 | cb_untilized | Untilized output (RM path only) | Compute (untilize) | Writer | Wt | Row |

### Kernel Arguments

**Compile-time** (compute kernel):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Tile-rows per core |
| 1 | Wt | uint32_t | Tiles in W dimension |
| 2 | input_is_rm | uint32_t | 1 if input is RM, 0 if TILE |
| 3 | has_gamma | uint32_t | 1 if gamma provided |

**Compile-time** (reader kernel):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * element_size (RM path) or tile_size (TILE path) |
| 1 | scaler_bits | uint32_t | 1.0f/W as float bits |
| 2 | eps_bits | uint32_t | epsilon as float bits |
| 3 | input_is_rm | uint32_t | 1 if RM, 0 if TILE |
| 4 | has_gamma | uint32_t | 1 if gamma provided |
| 5+ | TensorAccessorArgs | uint32_t[] | Input buffer accessor |

**Runtime** (reader kernel):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer address |
| 1 | start_id | uint32_t | Start stick (RM) or tile (TILE) ID |
| 2 | num_rows | uint32_t | Tile-rows for this core |
| 3 | gamma_addr | uint32_t | Gamma buffer address (0 if no gamma) |

**Compile-time** (writer kernel):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Output stick size (RM) or tile_size (TILE) |
| 1 | input_is_rm | uint32_t | 1 if RM output, 0 if TILE |
| 2+ | TensorAccessorArgs | uint32_t[] | Output buffer accessor |

**Runtime** (writer kernel):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | start_id | uint32_t | Start stick (RM) or tile (TILE) ID |
| 2 | num_rows | uint32_t | Tile-rows for this core |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count per phase
- [x] Reduce scaler CB (c_2) is bfloat16 format
- [x] DEST register holds max 8 tiles (bf16) / 4 tiles (f32); all helpers auto-chunk
- [x] RM CBs (c_0, c_17) count pages in tile-sized units for tilize/untilize compatibility
- [x] cb_tilized (c_1) holds Wt tiles for full-row persistence

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs PyTorch `x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps) * gamma`

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
| c_0 | Wt | Tile-sized pages (RM data or unused) | All (RM sticks) | Row |
| c_1 | Wt | TILE | All | Row (persistent) |
| c_2 | 1 | TILE (bf16) | Row0 of each face | Program |
| c_3 | 2 | TILE | All | Streaming |
| c_4 | 2 | TILE | Col0 (reduce_row output) | Row |
| c_5 | 1 | TILE | [0,0] (scalar eps) | Program |
| c_6 | 1 | TILE | Col0 (scalar per row) | Row |
| c_7 | Wt | TILE | All (or unused if no gamma) | Program |
| c_16 | Wt | TILE | All | Row |
| c_17 | Wt | Tile-sized (RM data, or unused) | All | Row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| square | MUL (self) | All (cb_tilized) | All (same) | NONE |
| add_eps_rsqrt | ADD | Col0 (cb_rms) | [0,0] (cb_eps) | SCALAR |
| normalize_mul | MUL | All (cb_tilized) | Col0 (cb_rms_inv) | COL |
| gamma_mul | MUL | All (cb_out) | All (cb_gamma, [1,Wt]) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | Reader, tilize, untilize, writer | identity | Same as input | - |
| 2 | square_reduce_rsqrt | Square, reduce, add_eps+rsqrt | rsqrt(mean(x^2)+eps) | Reduced W | [:,:,:,0:1] |
| 3 | normalize | mul_bcast_col (x * rms_inv) | rms_norm (no gamma) | Same as input | - |
| 4 | gamma | mul_bcast_row (out * gamma) | full rms_norm | Same as input | - |

### Stage 1: data_pipeline
- **Scope**: reader, writer, compute (tilize + identity copy + untilize for RM; passthrough for TILE)
- **Reference**: `input_tensor`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes c_0->c_1, copies c_1->c_16 tile-by-tile (copy_tile), untilizes c_16->c_17 for RM. For TILE, copies c_1->c_16.

### Stage 2: square_reduce_rsqrt
- **Scope**: compute kernel adds square, reduce, add_eps_rsqrt phases
- **Reference**: `torch.rsqrt(torch.mean(input_tensor**2, dim=-1, keepdim=True) + 1e-6)`
- **Output shape**: W dimension becomes 32 (one tile width, tile-aligned)
- **Compare slice**: `[:,:,:,0:1]` (only column 0 valid after REDUCE_ROW)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from stage 1**: Instead of copy c_1->c_16, compute does square(c_1)->c_3, reduce(c_3)->c_4, add_eps_rsqrt(c_4,c_5)->c_6, then copies c_6->c_16 for output. cb_tilized not popped until after mul phase (stage 3), but for this stage, pop after square since no mul yet.

### Stage 3: normalize
- **Scope**: compute kernel adds mul_bcast_col phase (x * rms_inv)
- **Reference**: `input_tensor * torch.rsqrt(torch.mean(input_tensor**2, dim=-1, keepdim=True) + 1e-6)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from stage 2**: After add_eps_rsqrt produces c_6, multiply c_1 (persisted) * c_6 (scalar bcast) -> c_16. Full output shape restored.

### Stage 4: gamma
- **Scope**: compute kernel adds gamma_mul phase; reader adds gamma loading and tilize
- **Reference**: `input_tensor * torch.rsqrt(torch.mean(input_tensor**2, dim=-1, keepdim=True) + 1e-6) * gamma`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from stage 3**: After normalize produces c_16, multiply c_16 * c_7 (gamma, row bcast) -> c_16 (in-place via intermediate CB swap).

### Reader Kernel

**RM path**: Batch-32-stick reading pattern from tilize reference. Pre-computes 32 NoC addresses, reads sticks into c_0, pushes Wt tile-pages. Before main loop: generates reduce scaler in c_2 via `prepare_reduce_scaler`, generates epsilon tile in c_5, reads+tilizes gamma sticks into c_7 (if gamma provided).

**TILE path**: Reads Wt tiles per row directly into c_1 via TensorAccessor. Same pre-loop scaler/epsilon/gamma generation.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_tilized, cb_scaler, cb_out)` with three-argument form (srcA=c_1, srcB=c_2, ocb=c_16).

**Pre-loop (if has_gamma and input_is_rm)**: Tilize gamma from c_0 -> c_7:
```cpp
compute_kernel_lib::tilize<c_0, c_7, InitAndUninit, WaitBlock, NoReconfigure>(Wt, 1);
```

**Main loop** (per tile-row, Ht iterations):

#### Phase 1: Tilize (RM path only)
```cpp
compute_kernel_lib::tilize<c_0, c_1, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(Wt, 1);
```
- In: c_0 [Wt tile-pages of RM sticks, from reader]
- Out: c_1 [Wt tiles, persistent for phases 2 and 5]

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| c_0 | 0 | freed (popped by tilize helper) |
| c_1 | Wt | pushed, persistent for reuse |

#### Phase 2: Square
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_1, c_3, BinaryInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, ALREADY PUSHED from Phase 1 or reader, WaitUpfrontNoPop - tiles persist]
- Out: c_3 [Wt tiles streamed, double-buffered]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | waited, not popped - persists for Phase 5 |
| c_3 | Wt | tiles streaming to reduce (may be partially consumed) |

#### Phase 3: Reduce SUM (mean via scaler=1/W) with rsqrt post_reduce_op
```cpp
compute_kernel_lib::reduce<
    PoolType::SUM,
    ReduceDim::REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_3, c_2, c_4,
    ReduceInputBlockShape::of(1, Wt, 1));
```
- In: c_3 [Wt tiles, streamed from square, WaitAndPopPerTile]
- Scaler: c_2 [1 tile, waited once internally, never popped]
- Out: c_4 [1 tile, mean(x^2) for this row]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | persists (not popped) |
| c_3 | 0 | all popped by reduce |
| c_4 | 1 | pushed (reduce output) |

#### Phase 4: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<
    BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_4, c_5, c_6, BinaryInputBlockShape::of(1, 1),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_4 [1 tile, reduce output, WaitAndPopPerTile]
- B: c_5 [1 tile, epsilon constant, WaitUpfrontNoPop - persists]
- Out: c_6 [1 tile, rsqrt(mean(x^2) + eps)]
- Post-op applies rsqrt inline

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | persists |
| c_4 | 0 | popped |
| c_5 | 1 | persists (never popped) |
| c_6 | 1 | pushed (rms_inv for this row) |

#### Phase 5: Normalize multiply (x * rms_inv, COL broadcast)
```cpp
compute_kernel_lib::mul<
    BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_1, c_6, c_16, BinaryInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop]
- B: c_6 [1 tile, rms_inv, WaitAndPopPerTile]. COL broadcast: 1 tile for Ht=1 row.
- Out: c_16 [Wt tiles, normalized output]

**CB state after Phase 5:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | still not popped |
| c_6 | 0 | popped by COL broadcast |
| c_16 | Wt | pushed |

Manual pop after Phase 5: `cb_pop_front(c_1, Wt)` to free for next row.

#### Phase 6: Gamma multiply (optional, ROW broadcast)
```cpp
if constexpr (has_gamma) {
    // Swap output through intermediate: c_16 -> process -> c_16
    // Use c_4 as intermediate (2 pages, reusable)
    compute_kernel_lib::mul<
        BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::WaitUpfrontNoPop,
        BinaryOutputPolicy::PerTile,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        c_16, c_7, c_4, BinaryInputBlockShape::of(1, Wt));
}
```

Note: c_4 is reused as gamma output (2 pages, double-buffered streaming). The writer reads from c_4 instead of c_16 when gamma is present. Alternatively, add a dedicated cb_gamma_out. For simplicity, the kernel writer should use c_4 as the gamma multiply output and adjust the writer source CB accordingly.

**Revised approach**: To avoid CB conflicts, when gamma is present the output goes to c_4 (reused). The untilize or writer then reads from c_4. When no gamma, reads from c_16.

#### Phase 7: Untilize (RM path only)
```cpp
compute_kernel_lib::untilize<Wt, cb_compute_out, c_17,
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::NoWait,
    untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
```
Where `cb_compute_out` is c_16 (no gamma) or c_4 (with gamma). Uses NoWait since tiles are already in the CB from the previous phase.

### Writer Kernel

**RM path**: Waits for Wt tile-pages in c_17 per row. Extracts 32 sticks via raw L1 pointer arithmetic. Writes each stick to DRAM via TensorAccessor with `noc_async_write`.

**TILE path**: Waits for tiles in c_16 (or c_4 with gamma) one at a time. Writes each tile to DRAM via TensorAccessor.

### Critical Notes
1. **cb_tilized (c_1) must persist**: Tiles are waited once in Phase 2 (WaitUpfrontNoPop) and reused in Phase 5 (NoWaitNoPop). Manual `cb_pop_front(c_1, Wt)` required after Phase 5.
2. **Scaler CB format**: c_2 MUST be bfloat16 regardless of input dtype. The reduce hardware handles format mismatch.
3. **Epsilon tile format**: c_5 should match input dtype for the add operation.
4. **Gamma tilize**: Gamma is always RM. Reader loads 32 sticks into c_0 before main loop. Compute tilizes c_0 -> c_7 before main loop.
5. **FP32 accumulation**: When input is float32, set `fp32_dest_acc_en=true` in ComputeConfig. This halves DEST capacity (4 tiles). All helpers auto-adapt via DEST_AUTO_LIMIT.
6. **ReconfigureRegisterDatatypeMode**: Each phase after the first needs INPUT_AND_OUTPUT reconfig since the CB formats change between operations. The helpers handle this via their reconfig template parameter.
7. **For TILE_LAYOUT path**: Reader writes directly to c_1. The compute kernel skips Phase 1 (tilize) and Phase 7 (untilize). Writer reads from c_16 (or c_4 with gamma).
8. **Per-row iteration**: The main loop in compute iterates Ht times. Each iteration processes one tile-row completely through all phases before starting the next row.

### Implementation Checklist
- [ ] Reader: RM stick batch-32 reading OR tile-by-tile reading, scaler/eps/gamma generation
- [ ] Compute: 7 phases max using helpers: tilize, square, reduce<SUM,REDUCE_ROW>, add+rsqrt, mul<COL>, mul<ROW>, untilize
- [ ] Writer: RM stick extraction OR tile-by-tile writing
- [ ] CB push/pop balance verified (c_1 manual pop after Phase 5)
