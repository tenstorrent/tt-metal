# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), batch_norm (compute_core)

## Mathematical Definition
```
mean[b,h] = (1/W) * sum_w(input[b,h,w])
var[b,h] = (1/W) * sum_w((input[b,h,w] - mean[b,h])^2)
standardized[b,h,w] = (input[b,h,w] - mean[b,h]) / sqrt(var[b,h] + epsilon)
output[b,h,w] = gamma[w] * standardized[b,h,w] + beta[w]   (if affine)
output[b,h,w] = standardized[b,h,w]                          (otherwise)
```
Row-wise layer normalization: normalize each row (last dimension) independently.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant |
| gamma | Tensor | No | shape (1,1,1,W) bf16 RM | None | Per-element scale |
| beta | Tensor | No | shape (1,1,1,W) bf16 RM | None | Per-element shift |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Layout | ROW_MAJOR_LAYOUT | Input must be row-major |
| Dtype | BFLOAT16 | Only bf16 supported |
| Memory | INTERLEAVED (DRAM) | Must be interleaved |
| Shape | >= 2D, H%32==0, W%32==0 | Dimensions must be tile-aligned |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR_LAYOUT
- **Memory**: INTERLEAVED (DRAM)

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce_row produces 1 tile per row, broadcast COL is scalar-like |
| gamma provided, beta absent | Apply scale only, skip bias add |
| gamma absent, beta provided | Skip scale, apply bias add only |
| Both absent | Pure standardization |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add gamma/beta RM stick reads, epsilon fill, reduce scaler generation |
| Compute | batch_norm + new | compute_core | Replace pre-computed stats with reduce_row for mean/var; tilize input, normalize, untilize output |
| Writer | untilize | output_stage | Extract RM sticks from untilized output CB, write to DRAM |

### Work Distribution
- **Work unit**: tile-row (one row of Wt tiles, corresponding to 32 RM sticks)
- **Grid**: 1D, up to `grid_size.x * grid_size.y` cores
- **Work per core**: `ceil(Ht_total / num_cores)` tile-rows, where `Ht_total = N*C*H / 32`
- **Remainder**: cliff core gets `Ht_total % tile_rows_per_core`

Uses `split_blocks_for_tilize(grid, Ht_total)` for work splitting.

### Data Flow
Reader reads 32 RM sticks per tile-row into c_0, plus gamma/beta sticks (once) and generates reduce scaler/epsilon tiles. Compute tilizes, reduces rows for mean, subtracts mean, squares, reduces for variance, adds epsilon, rsqrt, multiplies, optionally applies gamma/beta, then untilizes. Writer extracts RM sticks from output CB and writes to DRAM.

### Circular Buffer Requirements

Notation: `Wt = W / 32` (tiles per row).

| CB ID | Name | Purpose | Producer | Consumer | Pages | Page Size | Lifetime |
|-------|------|---------|----------|----------|-------|-----------|----------|
| c_0 | cb_input_rm | RM sticks from reader | Reader | Compute (tilize) | Wt | tile_size | Per tile-row |
| c_1 | cb_tilized | Tilized input | Compute | Compute | 2*Wt | tile_size | Per tile-row (double-buf for reuse) |
| c_2 | cb_gamma | Gamma RM sticks (tilized in compute) | Reader | Compute | Wt | tile_size | Program (once) |
| c_3 | cb_beta | Beta RM sticks (tilized in compute) | Reader | Compute | Wt | tile_size | Program (once) |
| c_8 | cb_reduce_scaler | Reduce scaler tile (1/W for AVG) | Reader | Compute | 1 | tile_size | Program |
| c_9 | cb_eps | Epsilon constant tile | Reader | Compute | 1 | tile_size | Program |
| c_16 | cb_out_rm | Untilized output RM sticks | Compute (untilize) | Writer | Wt | tile_size | Per tile-row |
| c_24 | cb_mean | Row-wise mean (Ht=1 tiles) | Compute | Compute | 1 | tile_size | Per tile-row |
| c_25 | cb_centered | Centered values (input - mean) | Compute | Compute | 2*Wt | tile_size | Per tile-row (double-buf for reuse) |
| c_26 | cb_sq | Squared centered values | Compute | Compute | Wt | tile_size | Per tile-row (streaming) |
| c_27 | cb_var | Variance tile | Compute | Compute | 1 | tile_size | Per tile-row |
| c_28 | cb_rstd | rsqrt(var + eps) tile | Compute | Compute | 1 | tile_size | Per tile-row |
| c_29 | cb_norm | Normalized output | Compute | Compute | Wt | tile_size | Per tile-row |
| c_30 | cb_scaled | After gamma multiply | Compute | Compute | Wt | tile_size | Per tile-row (only if gamma) |
| c_31 | cb_pre_untilize | Final tilized result before untilize | Compute | Compute (untilize) | Wt | tile_size | Per tile-row |

**CB routing for optional gamma/beta** (compile-time defines `HAS_GAMMA`, `HAS_BETA`):
- No gamma, no beta: untilize reads from cb_norm (c_29), writes to cb_out_rm (c_16)
- Gamma only: mul gamma -> cb_scaled (c_30), untilize from c_30
- Beta only: add beta -> cb_pre_untilize (c_31), untilize from c_31
- Both: mul gamma -> cb_scaled (c_30), add beta -> cb_pre_untilize (c_31), untilize from c_31

The compute kernel determines `cb_untilize_input` at compile time via defines.

### Kernel Arguments

**Reader Compile-Time Args:**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_input_rm | uint32_t | Input RM CB (c_0) |
| 1 | cb_gamma | uint32_t | Gamma CB (c_2) |
| 2 | cb_beta | uint32_t | Beta CB (c_3) |
| 3 | cb_reduce_scaler | uint32_t | Reduce scaler CB (c_8) |
| 4 | cb_eps | uint32_t | Epsilon CB (c_9) |
| 5 | stick_size | uint32_t | W * 2 bytes (bf16) |
| 6 | Wt | uint32_t | Tiles per row |
| 7 | has_gamma | uint32_t | 1 if gamma provided |
| 8 | has_beta | uint32_t | 1 if beta provided |
| 9+ | TensorAccessorArgs | varies | Input tensor accessor |
| +gamma | TensorAccessorArgs | varies | Gamma accessor (if has_gamma) |
| +beta | TensorAccessorArgs | varies | Beta accessor (if has_beta) |

**Reader Runtime Args:**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Input buffer base address |
| 1 | num_tile_rows | uint32_t | Tile-rows this core processes |
| 2 | start_page_id | uint32_t | First RM stick index |
| 3 | packed_eps | uint32_t | Epsilon packed as (bf16 << 16 \| bf16) |
| 4 | gamma_addr | uint32_t | Gamma buffer address (0 if absent) |
| 5 | beta_addr | uint32_t | Beta buffer address (0 if absent) |

**Compute Compile-Time Args:**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row |
| 1 | num_tile_rows | uint32_t | Tile-rows this core |
| 2 | has_gamma | uint32_t | 1 if gamma |
| 3 | has_beta | uint32_t | 1 if beta |

**Writer Compile-Time Args:**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_out_rm | uint32_t | Output RM CB (c_16) |
| 1 | stick_size | uint32_t | W * 2 bytes |
| 2 | Wt | uint32_t | Tiles per row |
| 3+ | TensorAccessorArgs | varies | Output tensor accessor |

**Writer Runtime Args:**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Output buffer base address |
| 1 | num_tile_rows | uint32_t | Tile-rows this core processes |
| 2 | start_page_id | uint32_t | First output stick index |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (helpers manage internally)
- [x] Reduce scaler CB (c_8) is bfloat16 (Float16_b)
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- helpers handle via DEST_AUTO_LIMIT
- [x] RM CBs count pages in sticks, tile CBs count in tiles -- c_0 uses tile_size pages but holds RM sticks (tilize pattern)

### Test Criteria

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile | Tile iteration | `(1, 1, 64, 128)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |
| Wide | Large W | `(1, 1, 32, 512)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (input_rm) | Wt | RM sticks in tile pages | All | Per tile-row |
| c_1 (tilized) | 2*Wt | TILE | All | Per tile-row (WaitUpfrontNoPop reuse) |
| c_2 (gamma) | Wt | RM sticks in tile pages | Row0 (1,1,1,W) | Program |
| c_3 (beta) | Wt | RM sticks in tile pages | Row0 (1,1,1,W) | Program |
| c_8 (reduce_scaler) | 1 | TILE (bf16) | Row0 | Program |
| c_9 (eps) | 1 | TILE (bf16) | Scalar | Program |
| c_16 (out_rm) | Wt | TILE (untilized RM data) | All | Per tile-row |
| c_24 (mean) | 1 | TILE | Col0 | Per tile-row |
| c_25 (centered) | 2*Wt | TILE | All | Per tile-row (WaitUpfrontNoPop reuse) |
| c_26 (sq) | Wt | TILE | All | Per tile-row (streaming) |
| c_27 (var) | 1 | TILE | Col0 | Per tile-row |
| c_28 (rstd) | 1 | TILE | Col0 | Per tile-row |
| c_29 (norm) | Wt | TILE | All | Per tile-row |
| c_30 (scaled) | Wt | TILE | All | Per tile-row (only if gamma) |
| c_31 (pre_untilize) | Wt | TILE | All | Per tile-row (only if beta) |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| sub_mean | SUB | All (c_1 tilized input) | Col0 (c_24 mean from REDUCE_ROW) | COL |
| square | MUL (self) | All (c_25 centered) | All (c_25 centered) | NONE (square) |
| add_eps | ADD | Col0 (c_27 variance) | Scalar (c_9 eps) | SCALAR |
| mul_rstd | MUL | All (c_25 centered) | Col0 (c_28 rstd) | COL |
| mul_gamma | MUL | All (c_29 norm) | All (c_2 gamma, 1D row) | ROW |
| add_beta | ADD | All (c_30 scaled) | All (c_3 beta, 1D row) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | Reader reads RM sticks; compute tilizes then untilizes (identity); writer writes RM sticks | `x` (identity passthrough) | Same as input | N/A |
| 2 | reduce_mean | Add REDUCE_ROW for mean computation; output = mean broadcast to input shape | `x.mean(dim=-1, keepdim=True).expand_as(x)` | Same as input | N/A |
| 3 | subtract_mean | Sub mean from input; output = centered values | `x - x.mean(dim=-1, keepdim=True)` | Same as input | N/A |
| 4 | variance_rsqrt | Square centered, reduce for var, add eps, rsqrt, multiply for standardization | `torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=1e-5)` | Same as input | N/A |
| 5 | affine | Apply gamma and beta | `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma.squeeze(), bias=beta.squeeze(), eps=1e-5)` | Same as input | N/A |

### Stage 1: data_pipeline
- **Scope**: reader.cpp, compute.cpp, writer.cpp -- full data movement pipeline
- **Reference**: `x` (identity)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`, `(1,1,32,512)`
- **Tolerances**: rtol=0.01, atol=0.01
- **Compute**: tilize c_0 -> c_1 (Wt tiles); copy_tiles c_1 -> c_31; untilize c_31 -> c_16
- **CB bypass**: Compute tilizes, copies through to cb_pre_untilize, untilizes to output. No normalization.

### Stage 2: reduce_mean
- **Scope**: compute.cpp adds reduce_row for mean, sub_bcast_col to write mean broadcast to output
- **Reference**: `x.mean(dim=-1, keepdim=True).expand_as(x)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`, `(1,1,32,512)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Add reduce scaler generation in reader. Add REDUCE_ROW in compute. Output mean broadcast via mul_bcast_col by rstd=1 (or copy_tiles of mean to full-width output). Actually, simplest: after reduce, broadcast mean via binary_op MUL with a ones-tile, or just use copy_tiles into output CB for each tile. For testability, output the mean expanded: use sub(input, mean, output) but negate, or better: just skip sub and output the mean broadcast across W using the add helper with broadcast COL, adding mean to a zero input. Simplest approach: compute tilizes input, reduces to get mean, then uses the `sub` helper with BroadcastDim::COL to compute (input - mean), then adds mean back: output = input. That tests nothing new. Better: output = mean broadcast. Compute tilizes, reduces to mean in c_24, then for each of Wt output tiles, copies the mean tile. This requires copy_tiles from c_24 to c_31 Wt times -- but c_24 has only 1 tile. Use NoWaitNoPop on c_24 to read it Wt times without popping, then pop after. Then untilize c_31 -> c_16.

### Stage 3: subtract_mean
- **Scope**: compute.cpp adds sub(input, mean) with COL broadcast
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`, `(1,1,32,512)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Replace mean-broadcast output with sub(tilized_input, mean) -> centered -> untilize

### Stage 4: variance_rsqrt
- **Scope**: compute.cpp adds square, reduce_row(var), add_eps, rsqrt, mul_rstd
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`, `(1,1,32,512)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: After sub_mean, square centered values, reduce_row for variance, add epsilon, rsqrt, multiply centered by rstd

### Stage 5: affine
- **Scope**: compute.cpp adds optional gamma multiply and beta add with ROW broadcast
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma.squeeze(), bias=beta.squeeze(), eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`, `(1,1,32,512)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Reader also reads gamma/beta RM sticks. Compute tilizes gamma/beta once, then applies mul_gamma and add_beta with ROW broadcast.

### Reader Kernel

Reads RM sticks from DRAM using TensorAccessor. Per tile-row: reserves Wt tile-pages in c_0, reads 32 sticks contiguously, pushes Wt pages. At program start: generates reduce scaler tile in c_8 via `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<c_8, AVG, REDUCE_ROW, 32, W>()`. Fills epsilon tile in c_9 using `dataflow_kernel_lib::prepare_reduce_scaler<c_9>(eps_float)` (packing epsilon as a constant tile with row0 fill -- suitable because add_tiles with SCALAR broadcast only uses [0,0]). If gamma present: reads W-width stick into c_2 (Wt tile-pages, 32 sticks -- only first stick has data, rest can be zero-padded). Same for beta into c_3. Gamma/beta are shape (1,1,1,W) so only 1 stick, but the tilize helper needs 32 sticks per block. The reader must zero-pad the remaining 31 sticks. Alternative: reader reads the single gamma/beta stick, and the compute kernel uses tilize with `total_input_pages=1` to handle the single-row case.

**Gamma/beta handling**: Since gamma/beta have shape (1,1,1,W), they have only 1 RM stick. The reader reserves Wt pages in c_2/c_3, reads 1 stick, zero-pads remaining 31*stick_size bytes, then pushes Wt pages. Compute tilizes this into a full tile-row where only row 0 has valid data. The binary_op MUL/ADD with BroadcastDim::ROW then broadcasts this single row across all 32 rows of each tile.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_8, c_16)` (three-arg: srcA=c_0, srcB=c_8, ocb=c_16)

Processing is per tile-row with an outer loop of `num_tile_rows`. Gamma/beta tilize happens once before the main loop.

#### Phase 0 (once): Tilize gamma/beta
If has_gamma: `compute_kernel_lib::tilize<Wt, c_2, cb_gamma_tilized>(1)` where cb_gamma_tilized reuses c_2 in-place (reader fills c_2 with RM sticks, compute tilizes c_2 -> a dedicated gamma tile CB). However, tilize requires different input and output CBs. Use c_2 as input, tile into c_29 temporarily, then the gamma tiles sit in c_29. But c_29 is also used for norm output. Solution: use separate CBs. Actually, for simplicity: reader puts gamma RM sticks into c_2, compute tilizes c_2 -> c_2 is consumed. We need a separate output CB for tilized gamma. Let us adjust: c_2 holds RM gamma sticks from reader; compute tilizes c_2 into a separate CB. But we are tight on CBs.

Revised plan: Reader puts gamma RM sticks into c_2. Compute tilizes c_2 -> c_30. Reader puts beta RM sticks into c_3. Compute tilizes c_3 -> c_31 if beta present. Then c_30 holds tilized gamma, c_31 holds tilized beta for the entire program. After the main loop, these are not popped -- they persist. The norm output goes to c_29. The untilize input depends on defines:
- No affine: untilize from c_29
- Gamma only: mul(c_29, c_30) -> c_31, untilize from c_31
- Beta only: add(c_29, c_3_tilized) -> c_31, untilize from c_31
- Both: mul(c_29, c_30) -> c_26 (reuse sq CB), add(c_26, c_31) -> c_29 (reuse), untilize from c_29

This gets complex. Simpler: use fixed routing.
- c_30 = tilized gamma (persists). c_31 = tilized beta (persists).
- After normalize -> c_29.
- If gamma: mul(c_29, c_30) -> c_26 (reuse). If also beta: add(c_26, c_31) -> cb_out_tile. Else: cb_out_tile = c_26.
- If no gamma but beta: add(c_29, c_31) -> cb_out_tile.
- If neither: cb_out_tile = c_29.
- untilize(cb_out_tile -> c_16).

Define `cb_out_tile` at compile time based on has_gamma and has_beta:
- Neither: cb_out_tile = c_29
- Gamma only: cb_out_tile = c_26 (after mul)
- Beta only: cb_out_tile = c_26 (after add)
- Both: cb_out_tile = c_29 (after mul into c_26, add into c_29 -- but c_29 is already consumed by mul read)

Actually, since binary_op helpers manage CB wait/pop internally, we can chain them cleanly. The key insight: use WaitUpfrontPopAtEnd for B operand on persistent gamma/beta tiles, and WaitAndPopPerTile for A.

Final CB routing (compile-time):
```
normalize_out = c_29
if has_gamma:
    mul(c_29, c_30, c_26)  -- gamma multiply, output to c_26
    affine_mid = c_26
else:
    affine_mid = c_29
if has_beta:
    add(affine_mid, c_31, c_29 or c_26)  -- beta add
    final_tile = c_26 if had gamma, c_26 if not (reuse)
else:
    final_tile = affine_mid
untilize(final_tile, c_16)
```

Simplified: always route through at most 2 extra CBs. The kernel writer will implement the conditional routing using compile-time defines.

#### Phase 1: Tilize input
```cpp
compute_kernel_lib::tilize<Wt, c_0, c_1>(1);  // 1 tile-row
```
- Input: c_0 [Wt pages of RM sticks, pushed by reader]
- Output: c_1 [Wt tilized tiles]
- InitUninitMode: InitOnly (first in chain)

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| c_0 | 0 | freed (tilize pops internally) |
| c_1 | Wt | freshly pushed, available for consumers |

#### Phase 2: Reduce row for mean
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop>(
    c_1, c_8, c_24,
    ReduceInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, FRESHLY PUSHED from Phase 1, WaitUpfrontNoPop -- tiles persist for Phase 3]
- B: c_8 [1 tile reduce scaler, pre-loaded by reader, persistent]
- Out: c_24 [1 tile, mean with valid data in Col0]
- Uses AVG scaler (1/W) in c_8, so SUM * (1/W) = mean.

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | waited, not popped -- persists for Phase 3 |
| c_24 | 1 | freshly pushed (mean tile) |

#### Phase 3: Subtract mean (centralize)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,     // A: c_1 already waited from Phase 2
    BinaryInputPolicy::WaitAndPopPerTile // B: c_24 mean, wait+pop per tile (1 tile reused via COL broadcast)
    >(c_1, c_24, c_25, BinaryInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop]
- B: c_24 [1 tile mean, COL broadcast across Wt tiles, WaitAndPopPerTile]
- Out: c_25 [Wt tiles, centered values]

Manual pop of c_1 after sub completes: `cb_pop_front(c_1, Wt)`.

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | must be manually popped after sub (NoWaitNoPop used) |
| c_24 | 0 | popped by sub helper (WaitAndPopPerTile on B with COL broadcast pops after all tiles processed) |
| c_25 | Wt | freshly pushed (centered values) |

After manual `cb_pop_front(c_1, Wt)`:
| c_1 | 0 | freed |

#### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop  // c_25 persists for Phase 7
    >(c_25, c_26, BinaryInputBlockShape::of(1, Wt));
```
- A: c_25 [Wt tiles, FRESHLY PUSHED from Phase 3, WaitUpfrontNoPop -- persists for Phase 7]
- Out: c_26 [Wt tiles, squared values]

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| c_25 | Wt | waited, not popped -- persists for Phase 7 |
| c_26 | Wt | freshly pushed (squared centered) |

#### Phase 5: Reduce row for variance
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    c_26, c_8, c_27,
    ReduceInputBlockShape::of(1, Wt));
```
- A: c_26 [Wt tiles, FRESHLY PUSHED from Phase 4, WaitAndPopPerTile]
- B: c_8 [1 tile reduce scaler, persistent]
- Out: c_27 [1 tile, variance with valid data in Col0]

**CB state after Phase 5:**
| CB | Tiles | State |
|----|-------|-------|
| c_26 | 0 | freed (popped by reduce helper) |
| c_27 | 1 | freshly pushed (variance tile) |

#### Phase 6: Add epsilon + rsqrt -> rstd
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,  // A: c_27 variance
    BinaryInputPolicy::NoWaitNoPop         // B: c_9 epsilon, persistent
    >(c_27, c_9, c_28, BinaryInputBlockShape::single(),
      [](uint32_t dst_idx) {
          rsqrt_tile_init();
          rsqrt_tile(dst_idx);
      });
```
- A: c_27 [1 tile variance, WaitAndPopPerTile]
- B: c_9 [1 tile epsilon, NoWaitNoPop -- persistent for entire program]
- Out: c_28 [1 tile, rsqrt(var + eps)]
- Post-op: rsqrt applied in-place in DEST before packing

**CB state after Phase 6:**
| CB | Tiles | State |
|----|-------|-------|
| c_27 | 0 | freed |
| c_28 | 1 | freshly pushed (rstd tile) |

#### Phase 7: Multiply centered by rstd (normalize)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,        // A: c_25 already waited from Phase 4
    BinaryInputPolicy::WaitAndPopPerTile   // B: c_28 rstd, COL broadcast
    >(c_25, c_28, c_29, BinaryInputBlockShape::of(1, Wt));
```
- A: c_25 [Wt tiles, ALREADY WAITED from Phase 4, NoWaitNoPop]
- B: c_28 [1 tile rstd, COL broadcast, WaitAndPopPerTile]
- Out: c_29 [Wt tiles, normalized output]

Manual pop of c_25 after mul completes: `cb_pop_front(c_25, Wt)`.

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_25 | Wt | must be manually popped |
| c_28 | 0 | freed |
| c_29 | Wt | freshly pushed (normalized) |

After manual `cb_pop_front(c_25, Wt)`:
| c_25 | 0 | freed |

#### Phase 8 (optional): Multiply by gamma
```cpp
if (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,  // A: c_29 normalized
        BinaryInputPolicy::NoWaitNoPop         // B: c_30 gamma, persistent
        >(c_29, c_30, c_26, BinaryInputBlockShape::of(1, Wt));
    // c_26 reused for gamma*norm output
}
```
- A: c_29 [Wt tiles normalized, WaitAndPopPerTile]
- B: c_30 [Wt tiles tilized gamma, NoWaitNoPop -- persistent, pre-waited before main loop]
- Out: c_26 [Wt tiles, scaled]

#### Phase 9 (optional): Add beta
```cpp
if (has_beta) {
    auto cb_in = has_gamma ? c_26 : c_29;
    compute_kernel_lib::add<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop
        >(cb_in, c_31, cb_final, BinaryInputBlockShape::of(1, Wt));
}
```
- A: c_26 or c_29 [Wt tiles, from previous phase]
- B: c_31 [Wt tiles tilized beta, NoWaitNoPop -- persistent]
- Out: cb_final [Wt tiles]

#### Phase 10: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_final, c_16>(1);
```
- Input: cb_final [Wt tiles, the final tilized result]
- Output: c_16 [Wt pages of RM sticks for writer]
- InitUninitMode: UninitOnly (last in chain)

### Writer Kernel

Waits for Wt tiles in c_16 per tile-row. Gets L1 read pointer, iterates 32 sticks. For each stick: computes L1 offset as `stick_idx * Wt * 32 * 2` bytes, gets NoC address via TensorAccessor, writes stick via `noc_async_write`. Barrier after 32 sticks, then `cb_pop_front(c_16, Wt)`.

### Critical Notes
1. **c_1 and c_25 require manual cb_pop_front** after sub and mul phases respectively, because NoWaitNoPop policy is used (tiles were pre-waited by earlier phases).
2. **Gamma/beta gamma_tilized in c_30/c_31 persist for entire program** -- waited once before main loop, popped once after main loop ends. The reader pushes them once; compute waits once. The binary_op calls use NoWaitNoPop for B operand on these CBs.
3. **c_8 (reduce scaler) and c_9 (epsilon) persist for entire program**. The compute kernel waits for c_8 at the start of reduce calls (handled by reduce helper). c_9 is waited manually before main loop and popped after.
4. **Reduce scaler for mean AND variance uses same value** (1/W) because both are AVG reductions over the W dimension. Single scaler tile in c_8 suffices.
5. **Gamma/beta tilize uses `total_input_pages=1`**: Since gamma/beta have shape (1,1,1,W) = only 1 RM stick, the reader writes 1 stick + 31 zero-padded sticks. Alternatively, use the asymmetric tilize mode: `tilize<Wt, c_2, c_30>(1, 1)` where `total_input_pages=1` tells tilize that only 1 input page exists. The reader would then push only 1 page. However, this asymmetric mode changes CB sync counts. Simpler: reader zero-pads to 32 sticks and uses standard tilize.

### Implementation Checklist
- [ ] Reader: RM stick reads with TensorAccessor, reduce scaler gen, epsilon fill, optional gamma/beta reads with zero-padding
- [ ] Compute: 10 phases using helpers: tilize, reduce(SUM, REDUCE_ROW) x2, sub(COL), square, add(SCALAR)+rsqrt post-op, mul(COL), optional mul(ROW), optional add(ROW), untilize
- [ ] Writer: Extract RM sticks from untilized CB, write to DRAM with TensorAccessor
- [ ] CB push/pop balance verified (manual pops for c_1, c_25 after NoWaitNoPop phases)
