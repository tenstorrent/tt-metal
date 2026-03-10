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
If gamma: output *= gamma[w]
If beta:  output += beta[w]
```
Layer normalization over the last dimension (W) of a row-major interleaved tensor.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Variance stabilizer |
| gamma | Tensor | No | shape (1,1,1,W), RM, bf16 | None | Per-element scale |
| beta | Tensor | No | shape (1,1,1,W), RM, bf16 | None | Per-element shift |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Dtype | BFLOAT16 | "Input must be bfloat16" |
| Layout | ROW_MAJOR | "Input must be row-major" |
| Memory | Interleaved | "Input must be interleaved" |
| Rank | >= 2 | "Input must be at least 2D" |
| H alignment | H % 32 == 0 | "Height must be tile-aligned (multiple of 32)" |
| W alignment | W % 32 == 0 | "Width must be tile-aligned (multiple of 32)" |

### Output Tensor Specification
- **Shape**: same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: Interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1; reduce is trivial (single tile) |
| gamma provided, beta None | Apply scale only |
| gamma None, beta provided | Apply shift only |
| gamma/beta width != W | RuntimeError |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add scaler/epsilon setup; optionally read gamma/beta |
| Compute (tilize) | tilize | input_stage | Use InitOnly mode (followed by more compute) |
| Compute (normalize) | batch_norm | compute_core | Replace pre-computed mean/var with reduce-based computation |
| Compute (untilize) | untilize | output_stage | Use UninitOnly or InitAndUninit at end |
| Writer | untilize | output_stage | RM stick extraction pattern |

### Work Distribution
- **Work unit**: tile-row (32 sticks across full width = Wt tiles)
- **Grid**: 1D linear, up to `compute_with_storage_grid_size()`
- **Work per core**: `nblocks_per_core = ceil(nblocks / ncores)` tile-rows
- **Remainder**: cliff core gets `nblocks % nblocks_per_core` tile-rows
- **nblocks**: `(N * C * H) / 32` = total tile-rows in tensor

### Data Flow
Reader reads 32 RM sticks per tile-row into c_0. Compute tilizes (c_0->c_1), normalizes through 7 phases in tile domain, then untilizes (c_final->c_16). Writer extracts 32 RM sticks from c_16 and writes to DRAM.

### Circular Buffer Requirements

Notation: Wt = W / 32 (tiles per row).

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input_rm | RM sticks for tilize | Reader | Compute | Wt | tile-row |
| c_1 | cb_tilized | Tilized input tiles | Compute | Compute | Wt | tile-row (persists through mean+sub) |
| c_8 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | program |
| c_9 | cb_eps | Epsilon tile | Reader | Compute | 1 | program |
| c_16 | cb_output_rm | Untilized RM output | Compute | Writer | Wt | tile-row |
| c_24 | cb_mean | Row mean (REDUCE_ROW output) | Compute | Compute | 1 | tile-row |
| c_25 | cb_centered | x - mean | Compute | Compute | Wt | tile-row (persists through square+mul) |
| c_26 | cb_centered_sq | (x - mean)^2 | Compute | Compute | Wt | tile-row |
| c_27 | cb_var | Row variance | Compute | Compute | 1 | tile-row |
| c_28 | cb_rsqrt | rsqrt(var + eps) | Compute | Compute | 1 | tile-row |
| c_31 | cb_normalized | Normalized output | Compute | Compute | Wt | tile-row |

**With gamma/beta (Stage 5 additions):**

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_2 | cb_gamma_rm | Gamma RM stick | Reader | Compute | 1 (stick-sized) | program |
| c_3 | cb_beta_rm | Beta RM stick | Reader | Compute | 1 (stick-sized) | program |
| c_29 | cb_gamma_tiled | Gamma tilized | Compute | Compute | Wt | program |
| c_30 | cb_beta_tiled | Beta tilized | Compute | Compute | Wt | program |

### Kernel Arguments

**Compile-time (Reader):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * sizeof(bfloat16) bytes |
| 1 | has_gamma | uint32_t | 1 if gamma provided |
| 2 | has_beta | uint32_t | 1 if beta provided |
| 3+ | TensorAccessorArgs(input) | uint32_t[] | Input tensor accessor |

**Runtime (Reader):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | num_sticks | uint32_t | nblocks_per_core * 32 |
| 2 | Wt | uint32_t | Tiles per row |
| 3 | scaler_value | uint32_t | float_to_uint32(1.0f / W) |
| 4 | eps_value | uint32_t | float_to_uint32(epsilon) |
| 5 | start_stick_id | uint32_t | First stick for this core |
| 6 | gamma_addr | uint32_t | Gamma buffer address (0 if absent) |
| 7 | beta_addr | uint32_t | Beta buffer address (0 if absent) |

**Compile-time (Compute):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row (compile-time for untilize template) |
| 1 | has_gamma | uint32_t | 1 if gamma provided |
| 2 | has_beta | uint32_t | 1 if beta provided |

**Runtime (Compute):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | nblocks | uint32_t | Tile-rows to process on this core |

**Compile-time (Writer):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * sizeof(bfloat16) bytes |
| 1 | Wt | uint32_t | Tiles per row |
| 2+ | TensorAccessorArgs(output) | uint32_t[] | Output tensor accessor |

**Runtime (Writer):**
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | nblocks | uint32_t | Tile-rows to process |
| 2 | start_stick_id | uint32_t | First output stick for this core |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB (c_8) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- Wt can exceed this; helpers auto-chunk
- [x] RM CBs count pages in sticks (c_2, c_3 only); tile CBs count in tiles (all others)
- [x] CB page sizes: c_2/c_3 use stick_size; all others use tile_size

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` (or manual formula)

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tests reduce across tiles | `(1, 1, 32, 128)` |
| Non-square | W != H | `(1, 1, 32, 256)` |
| Multi-row | Multiple tile-rows | `(1, 1, 64, 128)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Page Size | Layout | Valid Region | Lifetime |
|----|-------|-----------|--------|--------------|----------|
| c_0 | Wt | tile_size | RM (pre-tilize) | All | tile-row |
| c_1 | Wt | tile_size | TILE | All | tile-row |
| c_8 | 1 | tile_size | TILE | Row0 | program |
| c_9 | 1 | tile_size | TILE | All (scalar broadcast) | program |
| c_16 | Wt | tile_size | RM (post-untilize) | All | tile-row |
| c_24 | 1 | tile_size | TILE | Col0 (REDUCE_ROW output) | tile-row |
| c_25 | Wt | tile_size | TILE | All | tile-row |
| c_26 | Wt | tile_size | TILE | All | tile-row |
| c_27 | 1 | tile_size | TILE | Col0 | tile-row |
| c_28 | 1 | tile_size | TILE | Col0 -> All (via SCALAR bcast) | tile-row |
| c_31 | Wt | tile_size | TILE | All | tile-row |
| c_2 | 1 | stick_size | RM | Row0 | program (gamma) |
| c_3 | 1 | stick_size | RM | Row0 | program (beta) |
| c_29 | Wt | tile_size | TILE | Row0 | program (gamma tiled) |
| c_30 | Wt | tile_size | TILE | Row0 | program (beta tiled) |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 3 (sub mean) | SUB | All (c_1) | Col0 (c_24) | COL |
| 6 (add eps) | ADD | Col0 (c_27) | All (c_9) | SCALAR |
| 7 (mul rsqrt) | MUL | All (c_25) | Col0 (c_28) | COL |
| 8 (mul gamma) | MUL | All (c_31) | Row0 (c_29) | ROW |
| 9 (add beta) | ADD | All | Row0 (c_30) | ROW |

Note: Phase 6 A is Col0 and B is All-filled scalar -- SCALAR broadcast uses only [0,0] element of B, so the Add produces a Col0 result. The rsqrt post-op operates element-wise on the output tile in DEST. The result in c_28 has valid data only in Col0, which is correct for the subsequent COL broadcast in Phase 7.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader + tilize + untilize + Writer | output = input (passthrough) |
| 2 | subtract_mean | reduce_mean + sub | x - mean(x, dim=-1) |
| 3 | square_centered | square | (x - mean)^2 |
| 4 | full_normalize | reduce_var + add_eps_rsqrt + mul_rsqrt | layer_norm(x, eps) |
| 5 | gamma_beta | gamma/beta affine + reader changes | layer_norm(x, gamma, beta, eps) |

### Stage 1: data_pipeline
- **Scope**: reader, compute (tilize + untilize only), writer -- all 3 kernel files
- **Reference**: `x` (identity)
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes c_0->c_1, then immediately untilizes c_1->c_16. All normalize phases absent.
- **Compute pattern**: tilize<c_0, c_1>(Wt, 1) then untilize<Wt, c_1, c_16>(1). Reader sets up scaler/epsilon CBs but compute ignores them.

### Stage 2: subtract_mean
- **Scope**: compute kernel -- add reduce_mean (Phase 2) + sub_mean (Phase 3)
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from Stage 1**: Between tilize and untilize, add reduce+sub. Untilize now reads from c_25 instead of c_1.

### Stage 3: square_centered
- **Scope**: compute kernel -- add square (Phase 4)
- **Reference**: `(x - x.mean(dim=-1, keepdim=True)).pow(2)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from Stage 2**: Add square phase. Untilize reads from c_26 instead of c_25.

### Stage 4: full_normalize
- **Scope**: compute kernel -- add reduce_var (Phase 5) + add_eps_rsqrt (Phase 6) + mul_rsqrt (Phase 7)
- **Reference**: `(x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from Stage 3**: Add 3 new phases. Untilize reads from c_31.

### Stage 5: gamma_beta
- **Scope**: reader (gamma/beta loading), compute (tilize gamma/beta + Phases 8-9 affine)
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma.squeeze(), bias=beta.squeeze(), eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,32,256)`, `(1,1,64,128)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from Stage 4**: Reader reads gamma/beta sticks. Compute tilizes them (program-start), applies affine per tile-row. Untilize input CB changes based on gamma/beta presence.

### Reader Kernel
Reads 32 RM sticks per tile-row using TensorAccessor. At program start: fills c_8 (1/W scaler via `prepare_reduce_scaler`) and c_9 (epsilon via `fill_with_val`). Optionally reads gamma/beta single sticks into c_2/c_3. Main loop: `cb_reserve_back(c_0, Wt)`, read 32 sticks via noc_async_read, `cb_push_back(c_0, Wt)`.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_8, c_16)` (three-arg: srcA=c_0, srcB=c_8, ocb=c_16).

Epsilon wait: `cb_wait_front(c_9, 1)` once before main loop, `cb_pop_front(c_9, 1)` after.

**Per tile-row loop (nblocks iterations):**

#### Phase 1: Tilize (RM -> tile)
```cpp
compute_kernel_lib::tilize<c_0, c_1,
    tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);
```
- c_0: Wt pages RM data, waited and popped by helper
- c_1: Wt tiles pushed by helper

#### Phase 2: Reduce mean (row-wise sum * 1/W)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop>(
    c_1, c_8, c_24, ReduceInputBlockShape::row(Wt));
```
- c_1: Wt tiles waited upfront, NOT popped (persists for Phase 3)
- c_8: scaler 1/W, waited by helper, not popped (program lifetime)
- c_24: 1 tile mean pushed

#### Phase 3: Subtract mean (broadcast COL)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_1, c_24, c_25, BinaryInputBlockShape::of(1, Wt));
cb_pop_front(c_1, Wt);  // Manual pop -- NoWaitNoPop leaves c_1 tiles
```
- c_1: Wt tiles, NoWaitNoPop (already waited in Phase 2)
- c_24: 1 tile mean, WaitUpfrontPopAtEnd (consumed)
- c_25: Wt tiles centered, pushed
- After: manual cb_pop_front(c_1, Wt)

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | 0 | freed (manual pop) |
| c_24 | 0 | freed (PopAtEnd) |
| c_25 | Wt | freshly pushed |

#### Phase 4: Square centered
```cpp
compute_kernel_lib::square<BinaryInputPolicy::WaitUpfrontNoPop>(
    c_25, c_26, BinaryInputBlockShape::of(1, Wt));
```
- c_25: Wt tiles, WaitUpfrontNoPop (persists for Phase 7)
- c_26: Wt tiles centered^2, pushed

#### Phase 5: Reduce variance (row-wise sum * 1/W)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    c_26, c_8, c_27, ReduceInputBlockShape::row(Wt));
```
- c_26: Wt tiles, default WaitAndPopPerTile (consumed)
- c_27: 1 tile variance, pushed

#### Phase 6: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_27, c_9, c_28, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- c_27: 1 tile variance, consumed
- c_9: 1 tile epsilon, NoWaitNoPop (program lifetime, manually managed)
- c_28: 1 tile rsqrt(var+eps), pushed
- rsqrt applied as post_op in DEST before pack

#### Phase 7: Multiply by rsqrt (broadcast COL)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_25, c_28, c_31, BinaryInputBlockShape::of(1, Wt));
```
- c_25: Wt tiles centered, NoWaitPopAtEnd (already waited in Phase 4, pop at end)
- c_28: 1 tile rsqrt, WaitAndPopPerTile (COL broadcast, consumed)
- c_31: Wt tiles normalized, pushed

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_25 | 0 | freed (PopAtEnd) |
| c_28 | 0 | freed |
| c_31 | Wt | freshly pushed -- ready for untilize or affine |

#### Phase 8: Multiply gamma (optional, broadcast ROW)
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        c_31, c_29, c_25, BinaryInputBlockShape::of(1, Wt));
}
```
- c_31: Wt tiles normalized, consumed
- c_29: Wt tiles gamma (program lifetime, NoWaitNoPop)
- c_25: Wt tiles scaled, pushed

#### Phase 9: Add beta (optional, broadcast ROW)
```cpp
if constexpr (has_beta) {
    compute_kernel_lib::add<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        cb_affine_in, c_30, cb_affine_out, BinaryInputBlockShape::of(1, Wt));
}
```
Where `cb_affine_in`/`cb_affine_out` depend on gamma presence:
- gamma+beta: in=c_25, out=c_31
- beta only: in=c_31, out=c_25

#### Phase 10: Untilize (tile -> RM)
```cpp
compute_kernel_lib::untilize<Wt, cb_untilize_in, c_16,
    untilize_config::InitUninitMode::InitAndUninit>(1);
```
`cb_untilize_in` compile-time routing:
| has_gamma | has_beta | cb_untilize_in |
|-----------|----------|----------------|
| false | false | c_31 |
| true | false | c_25 |
| false | true | c_25 |
| true | true | c_31 |

### Writer Kernel
Per tile-row: `cb_wait_front(c_16, Wt)`, get L1 base via `get_read_ptr(c_16)`, write 32 sticks to DRAM using TensorAccessor (page_id = global stick index), barrier, `cb_pop_front(c_16, Wt)`.

### Critical Notes
1. **c_1 manual pop**: After Phase 3 sub, c_1 tiles must be manually popped (NoWaitNoPop on A does not pop).
2. **c_9 manual lifecycle**: Compute must `cb_wait_front(c_9, 1)` before main loop and `cb_pop_front(c_9, 1)` after.
3. **c_8 scaler persistence**: The reduce helper waits for c_8 internally but does not pop it. Caller must pop at program end.
4. **Wt as compile-time arg for untilize**: The untilize helper requires `block_width_tiles` as a compile-time template parameter.
5. **Gamma/beta tilize**: Uses asymmetric tilize with `total_input_pages=1` to handle single-row RM input.
6. **cb_untilize_in routing**: Factory sets the correct CB based on has_gamma/has_beta at compile time.

### Implementation Checklist
- [ ] Reader: TensorAccessor for RM sticks, prepare_reduce_scaler, fill_with_val for epsilon
- [ ] Compute: 10 phases using helpers: tilize, reduce(SUM, REDUCE_ROW), sub(COL), square, reduce(SUM, REDUCE_ROW), add(SCALAR)+rsqrt, mul(COL), mul(ROW), add(ROW), untilize
- [ ] Writer: TensorAccessor for RM sticks, stick extraction from untilized CB
- [ ] CB push/pop balance verified per tile-row iteration
