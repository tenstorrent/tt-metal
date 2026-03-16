# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input stage), untilize (output stage), batch_norm (compute core)

## Mathematical Definition
```
mean[i] = (1/W) * sum(input[i, :])
var[i]  = (1/W) * sum((input[i, :] - mean[i])^2)
output[i, j] = (input[i, j] - mean[i]) / sqrt(var[i] + eps)
if gamma: output[i, j] *= gamma[j]
if beta:  output[i, j] += beta[j]
```
Row-wise layer normalization: each row is independently normalized to zero mean and unit variance, with optional per-element affine transformation.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | 2D+, last 2 dims % 32 == 0 | - | Input tensor |
| gamma | Tensor | No | shape (1,1,1,W), RM, bf16 | None | Per-element scale |
| beta | Tensor | No | shape (1,1,1,W), RM, bf16 | None | Per-element shift |
| epsilon | float | No | > 0 | 1e-5 | Numerical stability |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | BFLOAT16 | "Input must be bfloat16" |
| layout | ROW_MAJOR | "Input must be ROW_MAJOR_LAYOUT" |
| memory | INTERLEAVED | "Input must be interleaved" |
| last dim | W % 32 == 0 | "Width must be tile-aligned (multiple of 32)" |
| 2nd-last dim | H % 32 == 0 | "Height must be tile-aligned (multiple of 32)" |
| gamma width | gamma.shape[-1] == W | "Gamma width must match input width" |
| beta width | beta.shape[-1] == W | "Beta width must match input width" |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize analysis | input_stage | Add gamma/beta RM reads, scaler/eps generation |
| Compute | batch_norm analysis + kernel_lib helpers | compute_core | Replace pre-computed stats with in-kernel reduce; add tilize/untilize phases |
| Writer | untilize analysis + existing writer | output_stage | Use `writer_unary_interleaved_start_id_blocked_rm_output.cpp` directly |

### Work Distribution
- **Work unit**: tile-row (32 consecutive rows spanning full tensor width)
- **Grid**: 1D (linearized from 2D device grid)
- **Total tile-rows**: `Ht_total = total_rows / 32` where `total_rows = product_of_all_dims_except_last`
- **Work per core**: `ceil(Ht_total / num_cores)` tile-rows
- **Remainder**: Last core (cliff) gets `Ht_total % tiles_per_core` tile-rows

### Data Flow
```
Reader (RISC-V 0, NOC0)          Compute (RISC-V 2)              Writer (RISC-V 1, NOC1)
========================          ====================             ========================
[Startup - once]
gen scaler -> cb_scaler           tilize gamma_rm -> cb_gamma
gen eps    -> cb_eps              tilize beta_rm  -> cb_beta
read gamma_rm -> cb_gamma_rm
read beta_rm  -> cb_beta_rm

[Per tile-row - Ht times]
read 32 sticks -> cb_in_rm        Phase 1: tilize(cb_in_rm -> cb_in)
                                  Phase 2: reduce_row(cb_in -> cb_mean)
                                  Phase 3: sub_col(cb_in - cb_mean -> cb_centered)
                                  Phase 4: square(cb_centered -> cb_sq)
                                  Phase 5: reduce_row(cb_sq -> cb_var)
                                  Phase 6: add_scalar(cb_var + cb_eps) + rsqrt -> cb_rsqrt
                                  Phase 7: mul_col(cb_centered * cb_rsqrt -> cb_norm)
                                  Phase 8: mul_row(cb_norm * cb_gamma -> cb_temp) [opt]
                                  Phase 9: add_row(cb_temp + cb_beta -> cb_out)  [opt]
                                  Phase 10: untilize(cb_out -> cb_out_rm)
                                                                  write 32 sticks <- cb_out_rm
```

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in_rm | RM input sticks | Reader | Compute | Wt | Per tile-row |
| c_1 | cb_in | Tilized input (persists for reduce+sub) | Compute | Compute | Wt | Phase 1-3 |
| c_2 | cb_scaler | Reduce scaler (1.0) | Reader | Compute | 1 | Program |
| c_3 | cb_mean | Row-wise mean | Compute | Compute | 1 | Phase 2-3 |
| c_4 | cb_centered | x - mean (persists for square+mul) | Compute | Compute | Wt | Phase 3-7 |
| c_5 | cb_sq | (x-mean)^2 | Compute | Compute | Wt | Phase 4-5 |
| c_6 | cb_var | Row-wise variance | Compute | Compute | 1 | Phase 5-6 |
| c_7 | cb_eps | Epsilon constant | Reader | Compute | 1 | Program |
| c_16 | cb_out | Final output tiles | Compute | Compute | Wt | Phase 7-10 |
| c_17 | cb_gamma | Tilized gamma (optional) | Compute | Compute | Wt | Program |
| c_18 | cb_beta | Tilized beta (optional) | Compute | Compute | Wt | Program |
| c_19 | cb_gamma_rm | RM gamma sticks (optional) | Reader | Compute | Wt | Startup |
| c_20 | cb_beta_rm | RM beta sticks (optional) | Reader | Compute | Wt | Startup |
| c_24 | cb_rsqrt | 1/sqrt(var+eps) | Compute | Compute | 1 | Phase 6-7 |
| c_25 | cb_temp | Scratch for affine routing | Compute | Compute | Wt | Phase 8-9 |
| c_28 | cb_out_rm | RM output for writer | Compute | Writer | Wt | Phase 10 |

All CBs use tile-sized pages (2048 bytes for bf16). Page counts are in tiles.

### Kernel Arguments

**Compile-time** (Reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * 2 bytes (one RM row) |
| 1 | Wt | uint32_t | Width in tiles = W / 32 |
| 2 | has_gamma | uint32_t | 1 if gamma provided |
| 3 | has_beta | uint32_t | 1 if beta provided |
| 4+ | TensorAccessorArgs(input) | auto | Input accessor |
| N+ | TensorAccessorArgs(gamma) | auto | Gamma accessor (slot always present) |
| M+ | TensorAccessorArgs(beta) | auto | Beta accessor (slot always present) |

**Runtime** (Reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | num_tile_rows | uint32_t | Tile-rows for this core |
| 2 | start_tile_row | uint32_t | Starting tile-row index |
| 3 | gamma_addr | uint32_t | Gamma buffer base (0 if absent) |
| 4 | beta_addr | uint32_t | Beta buffer base (0 if absent) |

**Compile-time** (Compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Width in tiles |
| 1 | block_size | uint32_t | min(Wt, 8) for tilize/untilize sub-blocks |
| 2 | has_gamma | uint32_t | 1 if gamma |
| 3 | has_beta | uint32_t | 1 if beta |

**Runtime** (Compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tile_rows | uint32_t | Tile-rows for this core |
| 1 | W | uint32_t | Width in elements (for 1/W scaling) |

**Compile-time** (Writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_size | uint32_t | Same as compute block_size |
| 1+ | TensorAccessorArgs(output) | auto | Output accessor |
| N | elem_size_bytes | uint32_t | 2 for bf16 |

**Runtime** (Writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | Wt | uint32_t | Width in tiles |
| 2 | num_tile_rows | uint32_t | Tile-rows for this core |
| 3 | start_tile_row | uint32_t | Starting tile-row index |
| 4 | H_logical | uint32_t | Total valid rows (clamp OOB writes) |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (reduce uses Wt; sub/mul/square use Wt)
- [x] Reduce scaler CB is bfloat16 (cb_scaler c_2)
- [x] DEST register holds max 8 tiles (bf16 half-sync); block_size = min(Wt, 8)
- [x] RM CBs count pages in tile-sized units (32 sticks = Wt tile pages)

### Test Criteria
- Output matches `torch.nn.functional.layer_norm(input, [W], weight=gamma, bias=beta, eps=eps)`
- Tolerances: rtol=0.05, atol=0.2 (multi-step computation)
- Test shapes:

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile-W | Tile iteration across width | `(1, 1, 32, 128)` |
| Multi-tile-HW | Multi-row + multi-col | `(1, 1, 64, 128)` |
| Non-square | W >> H | `(1, 1, 32, 256)` |
| Multi-batch | Batch dimension handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (cb_in_rm) | Wt | tile-sized, RM data | All | Per tile-row |
| c_1 (cb_in) | Wt | TILE | All | Phase 1-3 |
| c_2 (cb_scaler) | 1 | TILE (bf16) | Row0 | Program |
| c_3 (cb_mean) | 1 | TILE | Col0 | Phase 2-3 |
| c_4 (cb_centered) | Wt | TILE | All | Phase 3-7 |
| c_5 (cb_sq) | Wt | TILE | All | Phase 4-5 |
| c_6 (cb_var) | 1 | TILE | Col0 | Phase 5-6 |
| c_7 (cb_eps) | 1 | TILE | [0,0] | Program |
| c_16 (cb_out) | Wt | TILE | All | Phase 7-10 |
| c_17 (cb_gamma) | Wt | TILE | Row0 valid | Program |
| c_18 (cb_beta) | Wt | TILE | Row0 valid | Program |
| c_19 (cb_gamma_rm) | Wt | tile-sized, RM data | Row0 | Startup |
| c_20 (cb_beta_rm) | Wt | tile-sized, RM data | Row0 | Startup |
| c_24 (cb_rsqrt) | 1 | TILE | Col0 | Phase 6-7 |
| c_25 (cb_temp) | Wt | TILE | All | Phase 8-9 |
| c_28 (cb_out_rm) | Wt | tile-sized, RM data | All | Phase 10 |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 3 | SUB | All (cb_in) | Col0 (cb_mean) | COL |
| 6 | ADD | Col0 (cb_var) | [0,0] (cb_eps) | SCALAR |
| 7 | MUL | All (cb_centered) | Col0 (cb_rsqrt) | COL |
| 8 | MUL | All (cb_norm) | Row0 (cb_gamma) | ROW |
| 9 | ADD | All (cb_temp) | Row0 (cb_beta) | ROW |

Note: REDUCE_ROW produces Col0-valid output. Gamma/beta are (1,1,1,W) tilized with row 0 valid.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | pure_normalize | Full normalization (no gamma/beta): tilize, mean, center, var, rsqrt, normalize, untilize | `F.layer_norm(x, [shape[-1]], eps=1e-5)` | Same as input | - |
| 2 | gamma_scale | Add gamma multiplication | `F.layer_norm(x, [shape[-1]], weight=gamma, eps=1e-5)` | Same as input | - |
| 3 | full_affine | Add beta addition | `F.layer_norm(x, [shape[-1]], weight=gamma, bias=beta, eps=1e-5)` | Same as input | - |

### Stage 1: pure_normalize
- **Scope**: All 3 kernel files; compute phases 1-7 + 10
- **Reference**: `torch.nn.functional.layer_norm(input, [input.shape[-1]], eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **CB bypass**: gamma/beta phases (8-9) skipped; phase 7 output goes directly to cb_out

### Stage 2: gamma_scale
- **Scope**: Reader adds gamma read; compute adds phase 8
- **Reference**: `torch.nn.functional.layer_norm(input, [input.shape[-1]], weight=gamma, eps=1e-5)`
- **Delta from previous**: Reader reads gamma RM sticks at startup; compute tilizes gamma once, then applies mul<ROW> in phase 8
- **Shapes**: Same as stage 1
- **Tolerances**: rtol=0.05, atol=0.2

### Stage 3: full_affine
- **Scope**: Reader adds beta read; compute adds phase 9
- **Reference**: `torch.nn.functional.layer_norm(input, [input.shape[-1]], weight=gamma, bias=beta, eps=1e-5)`
- **Delta from previous**: Reader reads beta RM sticks; compute tilizes beta, applies add<ROW> in phase 9
- **Shapes**: Same as stage 1
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel

Reads RM sticks from DRAM. Uses TensorAccessor for all tensor accesses.

**Startup (once)**:
1. Generate reduce scaler: `dataflow_kernel_lib::prepare_reduce_scaler<c_2>(1.0f)` -> cb_scaler
2. Generate epsilon: `generate_bcast_unary_scalar(c_7, eps_packed)` -> cb_eps
3. If gamma: zero-fill cb_gamma_rm (Wt tiles), read 1 gamma stick at offset 0, push Wt pages
4. If beta: same pattern for cb_beta_rm

**Per tile-row**:
1. `cb_reserve_back(c_0, Wt)`
2. Read 32 sticks via TensorAccessor (page_id increments per stick)
3. `noc_async_read_barrier()`
4. `cb_push_back(c_0, Wt)`

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_1, c_2, c_16)` (srcA=cb_in, srcB=cb_scaler, dst=cb_out)

**Gamma/beta tilize (once, if present)**:
```cpp
tilize_all_blocks_to_cb<block_size>(c_19, c_17, Wt);  // gamma_rm -> gamma
tilize_all_blocks_to_cb<block_size>(c_20, c_18, Wt);  // beta_rm -> beta
```

**Per tile-row loop** (`num_tile_rows` iterations):

#### Phase 1: Tilize RM input
```cpp
tilize_all_blocks_to_cb<block_size>(c_0, c_1, Wt);
```
Uses `layernorm_compute_utils.h` (guarded by `#define TILIZE_IN`). Pops c_0, pushes Wt tiles to c_1.

#### Phase 2: Reduce row to mean
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop>(
    c_1, c_2, c_3, ReduceInputBlockShape::row(Wt),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [W](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, bit_cast<uint32_t>(1.0f / W));
    });
```
- A: c_1 [Wt tiles, FRESHLY PUSHED from Phase 1, **not popped** -- persists for Phase 3]
- Scaler: c_2 [1 tile, persistent]
- Out: c_3 [1 tile, mean with Col0 valid]

#### Phase 3: Subtract mean (center)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_1, c_3, c_4, BinaryInputBlockShape::of(1, Wt));
```
- A: c_1 [Wt tiles, ALREADY WAITED from Phase 2, popped at end]
- B: c_3 [1 tile, Col0 broadcast across Wt tiles, consumed]
- Out: c_4 [Wt tiles, centered values]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | 0 | freed (popped at end of sub) |
| c_3 | 0 | freed (consumed by sub) |
| c_4 | Wt | freshly pushed |

#### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_4, c_5, BinaryInputBlockShape::of(1, Wt));
```
- A: c_4 [Wt tiles, FRESHLY PUSHED from Phase 3, **not popped** -- persists for Phase 7]
- Out: c_5 [Wt tiles, squared values]

#### Phase 5: Reduce row to variance
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    c_5, c_2, c_6, ReduceInputBlockShape::row(Wt),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [W](uint32_t dst_idx) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst_idx, bit_cast<uint32_t>(1.0f / W));
    });
```
- A: c_5 [Wt tiles, streaming, consumed]
- Scaler: c_2 [1 tile, persistent]
- Out: c_6 [1 tile, variance with Col0 valid]

#### Phase 6: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR>(
    c_6, c_7, c_24, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_6 [1 tile, consumed]
- B: c_7 [1 tile, epsilon, persistent via WaitUpfrontNoPop]
- Out: c_24 [1 tile, 1/sqrt(var+eps) with Col0 valid]

Note: cb_eps (c_7) must use `WaitUpfrontNoPop` for B so it persists across tile-rows. The reader pushes cb_eps once at startup. On the first tile-row, the add helper waits for it. On subsequent tile-rows, it's already waited. Use `NoWaitNoPop` for B on subsequent iterations. Since this is in a loop, use `NoWaitNoPop` for B and handle the first-iteration wait externally: `cb_wait_front(c_7, 1)` before the main loop.

#### Phase 7: Multiply by rsqrt (normalize)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_4, c_24, cb_affine_or_out, BinaryInputBlockShape::of(1, Wt));
```
- A: c_4 [Wt tiles, ALREADY WAITED from Phase 4, popped at end]
- B: c_24 [1 tile, Col0 broadcast, consumed]
- Out: `cb_affine_or_out` -- routes to c_25 (if gamma or beta) or c_16 (if neither)

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_4 | 0 | freed |
| c_24 | 0 | freed |
| cb_affine_or_out | Wt | freshly pushed |

#### Phase 8 (optional): Multiply by gamma
```cpp
compute_kernel_lib::mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    cb_affine_or_out, c_17, cb_scaled_or_out, BinaryInputBlockShape::of(1, Wt));
```
- A: normalized tiles [Wt tiles, consumed]
- B: c_17 [Wt tiles, gamma, persistent -- NoWaitNoPop, never popped]
- Out: `cb_scaled_or_out` -- routes to c_25 (if beta) or c_16 (if no beta)

#### Phase 9 (optional): Add beta
```cpp
compute_kernel_lib::add<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    cb_scaled_or_out, c_18, c_16, BinaryInputBlockShape::of(1, Wt));
```
- A: scaled tiles [Wt tiles, consumed]
- B: c_18 [Wt tiles, beta, persistent -- NoWaitNoPop, never popped]
- Out: c_16 [Wt tiles, final output]

#### Dynamic CB routing for optional affine (same pattern as batch_norm):
```cpp
constexpr auto cb_affine_or_out = (has_gamma || has_beta) ? c_25 : c_16;
constexpr auto cb_scaled_or_out = has_beta ? c_25 : c_16;
```

#### Phase 10: Untilize output
```cpp
untilize_all_blocks_from_cb<block_size>(c_16, c_28, Wt);
```
Uses `layernorm_compute_utils.h` (guarded by `#define UNTILIZE_OUT`). Pops c_16, pushes Wt pages to c_28.

### Writer Kernel

Reuse the existing `writer_unary_interleaved_start_id_blocked_rm_output.cpp` from the layernorm directory. It:
1. Waits for untilized blocks in cb_out_rm (c_28)
2. Extracts 32 sticks per tile-row using blocked_range iterator
3. Writes each stick to DRAM via TensorAccessor with noc_async_write
4. Clamps writes to H_logical to avoid OOB DRAM writes

### Critical Notes

1. **cb_eps persistence**: Reader pushes cb_eps once. Compute must `cb_wait_front(c_7, 1)` before the main loop and `cb_pop_front(c_7, 1)` after all tile-rows complete. Phase 6 uses `NoWaitNoPop` for B.

2. **cb_gamma/cb_beta persistence**: Similarly waited once before main loop, never popped until program end.

3. **cb_scaler persistence**: Reduce helper handles scaler CB wait internally. The scaler tile persists across all reduce calls because the reduce helper with WaitAndPopPerTile on the scaler will wait and pop per reduce call. Since we call reduce twice per tile-row (mean, variance), the scaler needs 1 page and must NOT be popped by reduce. Use scaler with its own lifecycle: wait once before loop, pop after loop. The reduce helper's internal scaler handling should use the tile that's already waited.

4. **Gamma/beta tilize at startup**: The tilize calls for gamma/beta happen once before the main loop. The `tilize_all_blocks_to_cb` function does its own init/uninit, so it needs `reconfig_data_format` and hardware state restore afterward. The main loop's first `tilize_all_blocks_to_cb` call also does its own init.

5. **Gamma/beta reader pattern**: Reader zero-fills Wt tile pages (read from MEM_ZEROS), then writes 1 stick (W*2 bytes) at the base of the CB. This ensures rows 1-31 of each tile are zero after tilize, which is safe for ROW broadcast (only row 0 is read).

6. **block_size**: Set to `min(Wt, 8)`. Used by both tilize_all_blocks_to_cb and untilize_all_blocks_from_cb. Must be compile-time. Program factory should choose the largest divisor of Wt that is <= 8.

### Implementation Checklist
- [ ] Reader: RM stick reads with TensorAccessor, scaler/eps generation, optional gamma/beta RM reads with zero-fill
- [ ] Compute: 10 phases -- tilize, reduce(mean), sub(center), square, reduce(var), add+rsqrt, mul(normalize), mul(gamma), add(beta), untilize
- [ ] Writer: Reuse existing `writer_unary_interleaved_start_id_blocked_rm_output.cpp`
- [ ] CB push/pop balance verified per tile-row iteration
- [ ] Persistent CBs (scaler, eps, gamma, beta) managed outside main loop
