# Operation Design: layer_norm

## Overview
- **Operation Name**: layer_norm
- **Category**: normalization
- **Planning Mode**: Derivative
- **Reference Operation**: softmax W-small (`softmax_w_small_analysis.md`)

## Mathematical Definition
```
mu[h] = (1/W) * sum(x[h, w], w=0..W-1)           for each tile-row h
var[h] = (1/W) * sum((x[h, w] - mu[h])^2, w=0..W-1)
y[h, w] = (x[h, w] - mu[h]) / sqrt(var[h] + eps) * gamma[w] + beta[w]
```
Layer normalization over the W dimension. Each tile-row is normalized independently. Gamma/beta are optional per-element scale/shift tensors broadcast along H.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| eps | float | No | > 0 | 1e-5 | Numerical stability constant added to variance |
| W | uint32_t | Yes | multiple of 32 | - | Width dimension |
| H | uint32_t | Yes | multiple of 32 | - | Height dimension |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Shape | [1, 1, H, W] | Batch dims must be 1 for single-core |
| Layout | TILE | Must be tile layout |
| Dtype | BFLOAT16 | BFP16 only |
| Memory | DRAM interleaved | Must be interleaved |

Weight (gamma) and bias (beta): shape [1, 1, 1, W], TILE layout, BFP16, DRAM interleaved. Optional.

### Output Tensor Specification
- **Shape**: [1, 1, H, W] (same as input)
- **Dtype**: BFLOAT16
- **Layout**: TILE
- **Memory**: DRAM interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W == 32 (single tile per row) | Wt=1; reduce operates on single tile |
| gamma/beta absent | Skip scale/shift phases; output is just normalized x |
| H == 32 (single tile-row) | Ht=1; outer loop runs once |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | softmax W-small | input_stage | Add gamma/beta tile reads; scaler=1/W instead of 1.0 |
| Compute | softmax W-small | compute_core | Replace max+exp+recip with mean+variance+rsqrt+scale+shift |
| Writer | softmax W-small | output_stage | None |

### Work Distribution
- **Work unit**: tile-row (Wt tiles sharing same height-tile coordinate)
- **Grid**: single core (1x1)
- **Work per core**: Ht tile-rows
- **Remainder**: N/A (single core)

### Data Flow
Reader loads Wt input tiles per tile-row plus gamma/beta rows (once, persistent). Compute performs 6 sequential phases per tile-row: mean reduction, subtract mean, square differences, variance reduction with eps+rsqrt fused, multiply by rstd, then scale+shift. Writer drains Wt output tiles per tile-row.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input | Input tiles | Reader | Compute | Wt | Row: loaded once, read P1+P2, popped end P2 |
| c_1 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | Program: generated once, never popped |
| c_2 | cb_eps | Epsilon constant tile | Reader | Compute | 1 | Program: generated once, never popped |
| c_3 | cb_gamma | Weight tiles | Reader | Compute | Wt | Program: loaded once, never popped |
| c_4 | cb_beta | Bias tiles | Reader | Compute | Wt | Program: loaded once, never popped |
| c_16 | cb_output | Output tiles | Compute | Writer | Wt | Row: pushed P5/P6, drained by writer |
| c_24 | cb_mean | Mean scalar tile | Compute | Compute | 1 | Row: produced P1, consumed P2 |
| c_25 | cb_x_minus_mean | x - mean intermediate | Compute | Compute | Wt | Row: produced P2, consumed P3+P5 |
| c_26 | cb_var_eps_rsqrt | rsqrt(var + eps) scalar | Compute | Compute | 1 | Row: produced P4, consumed P5 |
| c_27 | cb_diff_sq | (x-mean)^2 intermediate | Compute | Compute | Wt | Row: produced P3, consumed P4 |

### Kernel Arguments

**Compile-time (Compute)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles |
| 1 | Wt | uint32_t | Width in tiles |
| 2 | has_gamma | uint32_t | 1 if weight provided |
| 3 | has_beta | uint32_t | 1 if bias provided |

**Compile-time (Reader)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles |
| 1 | Wt | uint32_t | Width in tiles |
| 2 | has_gamma | uint32_t | 1 if weight provided |
| 3 | has_beta | uint32_t | 1 if bias provided |
| 4+ | TensorAccessor args (input) | ... | Appended by TensorAccessorArgs |

**Runtime (Reader)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Input buffer base address |
| 1 | gamma_addr | uint32_t | Weight buffer base address (0 if absent) |
| 2 | beta_addr | uint32_t | Bias buffer base address (0 if absent) |
| 3 | eps_u32 | uint32_t | Bit-cast float eps value |

**Compile-time (Writer)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Height in tiles |
| 1 | Wt | uint32_t | Width in tiles |
| 2+ | TensorAccessor args (output) | ... | Appended by TensorAccessorArgs |

**Runtime (Writer)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Output buffer base address |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB (c_1) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- helpers auto-chunk
- [x] All CBs count pages in tiles (TILE layout throughout)

### Test Criteria
- Output shape matches input shape [1, 1, H, W]
- Numerical accuracy vs `torch.nn.functional.layer_norm`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile | Tile iteration | `(1, 1, 64, 128)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(1, 1, 128, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (input) | Wt | TILE | All [Ht, Wt] | Per-row |
| c_1 (scaler) | 1 | TILE | Row0 (scaler) | Program |
| c_2 (eps) | 1 | TILE | Scalar [0,0] | Program |
| c_3 (gamma) | Wt | TILE | Row0 [1, Wt] | Program |
| c_4 (beta) | Wt | TILE | Row0 [1, Wt] | Program |
| c_16 (output) | Wt | TILE | All [Ht, Wt] | Per-row |
| c_24 (mean) | 1 | TILE | Col0 (reduce output) | Per-row |
| c_25 (x-mean) | Wt | TILE | All | Per-row |
| c_26 (rstd) | 1 | TILE | Col0 (reduce output) | Per-row |
| c_27 (diff_sq) | Wt | TILE | All | Per-row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| P2: x - mean | SUB | All (c_0) | Col0 (c_24: REDUCE_ROW output) | COL |
| P3: (x-mean)^2 | SQUARE | All (c_25) | same | N/A |
| P5: (x-mean)*rstd | MUL | All (c_25) | Col0 (c_26: REDUCE_ROW output) | COL |
| P5b: *gamma | MUL | All (c_16 interim) | Row0 (c_3: [1,Wt]) | ROW |
| P5c: +beta | ADD | All (c_16 interim) | Row0 (c_4: [1,Wt]) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader+Writer pass-through, compute identity copy | `x` (input passthrough) |
| 2 | subtract_mean | P1 (mean reduce) + P2 (subtract mean) | `x - x.mean(dim=-1, keepdim=True)` |
| 3 | normalize | P3 (square) + P4 (var reduce + eps + rsqrt) + P5 (multiply rstd) | `(x - x.mean(-1, True)) / torch.sqrt(x.var(-1, True, False) + 1e-5)` |
| 4 | scale_shift | P5b (gamma) + P5c (beta) | `torch.nn.functional.layer_norm(x, [W], weight=gamma, bias=beta, eps=1e-5)` |

### Stage 1: data_pipeline
- **Scope**: reader.cpp, writer.cpp, compute.cpp (identity: cb_input -> cb_output copy via sub helper with zero)
- **Reference**: `x` (identity)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(1,1,128,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute copies input tiles directly to output using `sub<COL>` with a zero-valued scalar CB (c_24), effectively `x - 0 = x`. Reader generates scaler CB but compute uses only the identity path. No gamma/beta reads.

### Stage 2: subtract_mean
- **Scope**: compute.cpp adds P1 (reduce mean) + P2 (subtract)
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Delta from previous**: Replace identity copy with reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop> + sub<COL>
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(1,1,128,64)`
- **Tolerances**: rtol=0.02, atol=0.1

### Stage 3: normalize
- **Scope**: compute.cpp adds P3 (square) + P4 (var reduce+rsqrt) + replaces P5
- **Reference**: `(x - x.mean(-1, True)) / torch.sqrt(x.var(-1, True, False) + 1e-5)`
- **Delta from previous**: After subtract_mean, square differences, reduce to variance, fuse eps+rsqrt as post-reduce op, multiply x_minus_mean by rstd
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(1,1,128,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Stage 4: scale_shift
- **Scope**: compute.cpp adds P5b (mul gamma) + P5c (add beta); reader adds gamma/beta loading
- **Reference**: `torch.nn.functional.layer_norm(x, [W], weight=gamma, bias=beta, eps=1e-5)`
- **Delta from previous**: After normalize, multiply output by gamma (ROW broadcast), add beta (ROW broadcast)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(1,1,128,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel
Before main loop: generate cb_scaler (1/W) via `prepare_reduce_scaler`, generate cb_eps (eps value) via `prepare_reduce_scaler`. If has_gamma, read Wt tiles into cb_gamma. If has_beta, read Wt tiles into cb_beta. Main loop (Ht iterations): read Wt input tiles into cb_input using TensorAccessor.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input, cb_scaler, cb_output)`

Wait once for persistent CBs: `cb_wait_front(cb_scaler, 1)`, `cb_wait_front(cb_eps, 1)`. If has_gamma: `cb_wait_front(cb_gamma, Wt)`. If has_beta: `cb_wait_front(cb_beta, Wt)`.

Main loop runs Ht times (one per tile-row):

#### Phase 1: Mean reduction
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_input, cb_scaler, cb_mean,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- A: cb_input [Wt tiles, FRESHLY PUSHED by reader, NoPop -- tiles persist for P2]
- B: cb_scaler [1 tile, program-lifetime, never popped]
- Out: cb_mean [1 tile, pushed by reduce helper]

Scaler value = 1/W, so `reduce<SUM>` with scaler=1/W computes mean directly.

#### Phase 2: Subtract mean (x - mean)
```cpp
compute_kernel_lib::sub<compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    cb_input, cb_mean, cb_x_minus_mean,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_input [Wt tiles, ALREADY WAITED from P1 via WaitUpfrontNoPop, NoWaitNoPop policy]
- B: cb_mean [1 tile, FRESHLY PUSHED by P1, popped by helper (WaitAndPopPerTile on COL = 1 pop per row)]
- Out: cb_x_minus_mean [Wt tiles]

Manual `cb_pop_front(cb_input, Wt)` after this call to release input tiles.

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| cb_input | 0 | freed (manually popped) |
| cb_mean | 0 | freed (popped by sub helper) |
| cb_x_minus_mean | Wt | freshly pushed |

#### Phase 3: Square differences
```cpp
compute_kernel_lib::square<
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_x_minus_mean, cb_diff_sq,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_x_minus_mean [Wt tiles, FRESHLY PUSHED by P2, NoPop -- tiles persist for P5]
- Out: cb_diff_sq [Wt tiles]

#### Phase 4: Variance reduction + eps + rsqrt
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
    cb_diff_sq, cb_scaler, cb_var_eps_rsqrt,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::NoAccumulation{},
    [](uint32_t dst_idx) {
        // Add eps: cb_eps tile has eps in scaler position
        // Use add_tiles_bcast to add eps scalar to variance
        // Then rsqrt
    });
```

The post-reduce lambda for fusing eps addition and rsqrt is complex. The eps addition cannot trivially use `add_tiles_bcast` inside a lambda since that requires CB operations inside tile_regs scope. Instead, we use a two-step approach:

**Step 4a**: Reduce variance with `WaitAndPopPerTile` (frees cb_diff_sq tiles):
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
    cb_diff_sq, cb_scaler, cb_var_eps_rsqrt,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```

**Step 4b**: Add eps + rsqrt via binary add with post-op:
```cpp
compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::SCALAR,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
    cb_var_eps_rsqrt, cb_eps, cb_var_eps_rsqrt_final,
    compute_kernel_lib::BinaryInputBlockShape::single(), {},
    compute_kernel_lib::NoAccumulation{},
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```

This requires a second 1-tile CB for the rsqrt output. Revise: use cb_24 (cb_mean, which is free after P2) as the final rstd output, and cb_26 as intermediate variance.

**Revised CB assignment for P4**:
- cb_diff_sq reduced into cb_26 (intermediate variance, 1 tile)
- add eps + rsqrt from cb_26 into cb_24 (reuse cb_mean slot, now cb_rstd)

Updated names: c_24 serves dual purpose (cb_mean in P1-P2, cb_rstd in P4-P5). c_26 = cb_variance (intermediate).

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| cb_x_minus_mean (c_25) | Wt | persists from P2 (NoPop in P3) |
| cb_diff_sq (c_27) | 0 | freed by reduce in P4a |
| cb_variance (c_26) | 0 | freed by add in P4b |
| cb_rstd (c_24) | 1 | freshly pushed (rsqrt(var+eps)) |

#### Phase 5: Multiply by rstd
```cpp
compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    cb_x_minus_mean, cb_rstd, cb_output,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_x_minus_mean [Wt tiles, persisted from P2, PopAtEnd -- freed after multiply]
- B: cb_rstd [1 tile, FRESHLY PUSHED by P4b, popped per row by COL broadcast]
- Out: cb_output [Wt tiles]

#### Phase 5b (conditional: has_gamma): Scale by gamma
Requires reading output back. Since cb_output is consumed by the writer, we need to output to a temporary and then apply gamma+beta. Alternative: output P5 to cb_output, have writer not drain yet, then overwrite.

Better approach: P5 outputs to cb_diff_sq (c_27, now free, reused as temp), then P5b reads from c_27 and applies gamma, outputting to cb_output (or another temp if beta follows).

**Revised flow for scale+shift**:
- P5 outputs to c_27 (temp_norm, Wt tiles)
- P5b: mul<ROW>(c_27, cb_gamma, c_25_reused) -- but c_25 still has data from P5? No, P5 uses WaitUpfrontPopAtEnd on c_25, so it is freed.
- Actually, P5 with PopAtEnd on c_25 means c_25 is freed after P5. So:
  - P5 outputs normalized result to c_27 (reuse diff_sq, free since P4a)
  - P5b: if has_gamma && has_beta: mul<ROW>(c_27, cb_gamma, c_25) with PopAtEnd on c_27
  - P5c: add<ROW>(c_25, cb_beta, cb_output) with PopAtEnd on c_25
  - If has_gamma only: mul<ROW>(c_27, cb_gamma, cb_output)
  - If has_beta only: add<ROW>(c_27, cb_beta, cb_output)
  - If neither: copy c_27 -> cb_output... or just output P5 directly to cb_output.

**Final revised flow**:
- If no gamma and no beta: P5 outputs directly to cb_output.
- If gamma or beta present: P5 outputs to c_27, then conditionally apply gamma and/or beta, final result to cb_output.

For simplicity in the kernel, always output P5 to c_27 when has_gamma || has_beta, otherwise to cb_output.

#### Phase 5b: Scale by gamma (conditional)
```cpp
if (has_gamma) {
    compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::ROW,
        compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
        compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
        cb_temp_norm, cb_gamma, has_beta ? cb_x_minus_mean : cb_output,
        compute_kernel_lib::BinaryInputBlockShape::row(Wt));
}
```
- A: cb_temp_norm (c_27) [Wt tiles, PopAtEnd -- freed]
- B: cb_gamma (c_3) [Wt tiles, program-lifetime, NoWaitNoPop -- persists]
- Out: cb_x_minus_mean (c_25, reused) if beta follows, else cb_output (c_16)

#### Phase 5c: Add beta (conditional)
```cpp
if (has_beta) {
    uint32_t src_cb = has_gamma ? cb_x_minus_mean : cb_temp_norm;
    compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::ROW,
        compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
        compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
        src_cb, cb_beta, cb_output,
        compute_kernel_lib::BinaryInputBlockShape::row(Wt));
}
```
- A: source [Wt tiles, PopAtEnd -- freed]
- B: cb_beta (c_4) [Wt tiles, program-lifetime, NoWaitNoPop -- persists]
- Out: cb_output (c_16) [Wt tiles]

### Writer Kernel
Main loop (Ht iterations): wait for Wt tiles in cb_output, write to DRAM via TensorAccessor, pop Wt.

### Critical Notes
1. **cb_input manual pop**: After P2 (sub with NoWaitNoPop on A), manually `cb_pop_front(cb_input, Wt)` since the reduce helper in P1 used WaitUpfrontNoPop and the sub helper uses NoWaitNoPop -- neither pops cb_input.
2. **CB reuse**: c_24 serves as cb_mean (P1-P2) then cb_rstd (P4b-P5). c_27 serves as cb_diff_sq (P3-P4a) then cb_temp_norm (P5). c_25 serves as cb_x_minus_mean (P2-P5) then as intermediate for gamma*x when both gamma and beta are present. All reuses are safe because prior data is consumed before reuse.
3. **Scaler value**: cb_scaler contains 1/W (not 1.0). This makes `reduce<SUM>` compute the mean directly and the variance directly, without needing a separate division.
4. **Gamma/beta ROW broadcast**: gamma is [1, Wt] so ROW broadcast replicates each column's weight across all rows of the tile. This is correct for per-feature normalization.
5. **Eps as SCALAR broadcast**: cb_eps contains the eps value in scaler position. The add<SCALAR> in P4b broadcasts it to match variance tile shape.

### Implementation Checklist
- [ ] Reader: TensorAccessor reads, `prepare_reduce_scaler` for scaler (1/W) and eps CBs, gamma/beta tile loads
- [ ] Compute: 6 phases using helpers: reduce (P1, P4a), sub (P2), square (P3), add+rsqrt (P4b), mul (P5, P5b), add (P5c)
- [ ] Writer: TensorAccessor writes, Wt tiles per iteration
- [ ] CB push/pop balance verified per tile-row iteration
