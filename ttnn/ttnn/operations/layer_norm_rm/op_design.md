# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), softmax (compute_core)

## Mathematical Definition
```
mean[b,c,h] = sum(x[b,c,h,:]) / W
centered[b,c,h,w] = x[b,c,h,w] - mean[b,c,h]
var[b,c,h] = sum(centered[b,c,h,:]^2) / W
output[b,c,h,w] = centered[b,c,h,w] * rsqrt(var[b,c,h] + epsilon)
if gamma: output *= gamma[w]
if beta:  output += beta[w]
```
Layer normalization over the last dimension (W) of a row-major interleaved tensor. Tilize/untilize happen in-kernel.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Variance stabilizer |
| gamma | Tensor | No | shape (1,1,1,W) bf16 RM | None | Per-element scale |
| beta | Tensor | No | shape (1,1,1,W) bf16 RM | None | Per-element shift |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | BFLOAT16 | "layer_norm_rm only supports bfloat16" |
| layout | ROW_MAJOR | "layer_norm_rm requires ROW_MAJOR layout" |
| memory | interleaved DRAM | "requires interleaved memory config" |
| rank | >= 2 | "tensor must be at least 2D" |
| last 2 dims | aligned to 32 | "H and W must be multiples of 32" |

### Output Tensor Specification
- **Shape**: same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce is trivial |
| gamma=None, beta=None | Skip scale/shift phases |
| gamma only | Apply scale, skip shift |
| epsilon very small | Numerical instability possible, user responsibility |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add 3-pass row streaming, gamma/beta tile reads |
| Compute (tilize) | tilize | input_stage | None |
| Compute (normalize) | softmax | compute_core | Replace max/exp/sum with mean/var/rsqrt; use helpers |
| Compute (untilize) | untilize | output_stage | None |
| Writer | untilize | output_stage | None |

### Work Distribution
- **Work unit**: tile-row (32 rows x full width = Wt tiles)
- **Grid**: 1D linear, up to `grid_size.x * grid_size.y` cores
- **Work per core**: `nblocks_per_core` tile-rows via `split_blocks_for_tilize`
- **Remainder**: cliff core gets `nblocks % nblocks_per_core`

### Data Flow

The reader sends each tile-row 3 times (3-pass streaming). Pass 1: compute mean. Pass 2: compute variance. Pass 3: normalize + optional gamma/beta. Tilize converts RM sticks to tiles on each pass. Untilize converts the final normalized tiles back to RM sticks. Writer drains RM sticks to DRAM.

### Circular Buffer Requirements

**Notation**: Wt = W/32 (tiles per row), Ht = total tile-rows assigned to this core.

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input_rm | RM sticks from reader | Reader | Compute(tilize) | Wt | Block |
| c_1 | cb_tilized | Tilized input tiles | Compute(tilize) | Compute(ops) | Wt | Block |
| c_2 | cb_reduce_scaler | Scaler for reduce (1/W as bf16) | Reader | Compute | 1 | Program |
| c_3 | cb_eps_scalar | Epsilon tile (bf16) | Reader | Compute | 1 | Program |
| c_4 | cb_gamma | Gamma tiles (1 row, Wt tiles) | Reader | Compute | Wt | Program |
| c_5 | cb_beta | Beta tiles (1 row, Wt tiles) | Reader | Compute | Wt | Program |
| c_24 | cb_mean | Row-reduced mean (1 tile, col vector) | Compute | Compute | 1 | Row (pass1->pass2->pass3) |
| c_25 | cb_centered | Centered tiles (x - mean) | Compute | Compute | Wt | Block (within pass2) |
| c_26 | cb_var | Row-reduced variance (1 tile, col vector) | Compute | Compute | 1 | Row (pass2) |
| c_27 | cb_rsqrt_var | rsqrt(var+eps) (1 tile, col vector) | Compute | Compute | 1 | Row (pass3) |
| c_28 | cb_normalized | Normalized output tiles | Compute | Compute | Wt | Block (within pass3) |
| c_16 | cb_output_tiles | Pre-untilize output (tiled) | Compute | Compute(untilize) | Wt | Block |
| c_17 | cb_output_rm | Untilized RM output | Compute(untilize) | Writer | Wt | Block |

### Kernel Arguments

**Compile-time** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | Bytes per RM stick (W * 2) |
| Reader | 1+ | input_accessor_args | varies | TensorAccessorArgs for input |
| Compute | 0 | Wt | uint32_t | Tiles per row |
| Compute | 1 | nblocks | uint32_t | Tile-rows for this core |
| Compute | 2 | has_gamma | uint32_t | 1 if gamma present |
| Compute | 3 | has_beta | uint32_t | 1 if beta present |
| Writer | 0 | cb_id_out | uint32_t | Output CB index (c_17) |
| Writer | 1 | output_stick_size | uint32_t | Bytes per output RM stick |
| Writer | 2+ | output_accessor_args | varies | TensorAccessorArgs for output |

**Runtime** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | input_addr | uint32_t | Input buffer DRAM address |
| Reader | 1 | start_stick_id | uint32_t | First stick for this core |
| Reader | 2 | num_sticks | uint32_t | Total sticks (nblocks * 32) |
| Reader | 3 | gamma_addr | uint32_t | Gamma buffer addr (0 if none) |
| Reader | 4 | beta_addr | uint32_t | Beta buffer addr (0 if none) |
| Writer | 0 | output_addr | uint32_t | Output buffer DRAM address |
| Writer | 1 | start_stick_id | uint32_t | First output stick for this core |
| Writer | 2 | num_sticks | uint32_t | Total sticks to write |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (Wt for data CBs, 1 for scalar CBs)
- [x] Reduce scaler CB (c_2) is bfloat16
- [x] DEST register holds max 4 tiles (bf16 with fp32_dest_acc_en) -- helpers handle DEST limits automatically
- [x] RM CBs count pages in tiles (tilize convention: 32 sticks = Wt tile-pages)

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
| c_0 (input_rm) | Wt | RM-as-tile | All | Block (per pass) |
| c_1 (tilized) | Wt | Tile | All | Block (per pass) |
| c_2 (reduce_scaler) | 1 | Tile, bf16 | Row0 | Program |
| c_3 (eps_scalar) | 1 | Tile, bf16 | Scalar[0,0] | Program |
| c_4 (gamma) | Wt | Tile | Row0 (1D broadcast) | Program |
| c_5 (beta) | Wt | Tile | Row0 (1D broadcast) | Program |
| c_24 (mean) | 1 | Tile | Col0 (REDUCE_ROW output) | Row |
| c_25 (centered) | Wt | Tile | All | Block |
| c_26 (var) | 1 | Tile | Col0 | Row |
| c_27 (rsqrt_var) | 1 | Tile | Col0 | Row |
| c_28 (normalized) | Wt | Tile | All | Block |
| c_16 (output_tiles) | Wt | Tile | All | Block |
| c_17 (output_rm) | Wt | RM-as-tile | All | Block |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| subtract_mean | SUB | All (c_1 tilized) | Col0 (c_24 mean) | COL |
| square | SQUARE | All (c_25 centered) | same | NONE |
| add_eps | ADD | Col0 (c_26 var) | Scalar[0,0] (c_3 eps) | SCALAR |
| mul_rsqrt | MUL | All (c_1 tilized re-centered) | Col0 (c_27 rsqrt_var) | COL |
| apply_gamma | MUL | All (c_28 normalized) | Row0 (c_4 gamma) | ROW |
| apply_beta | ADD | All (result) | Row0 (c_5 beta) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | Reader+tilize+untilize+writer (identity) | input passthrough | Same as input | N/A |
| 2 | reduce_mean | Pass1: tilize, reduce_sum_row, scaler=1/W | Row-wise mean | Tile-aligned col vector | `[:,:,:,0:1]` |
| 3 | subtract_mean | Pass1 mean + pass2 sub(x, mean) | x - mean(x) | Same as input | N/A |
| 4 | variance_rsqrt | Pass2: square, reduce, add_eps, rsqrt | rsqrt(var+eps) col vector | Tile-aligned col vector | `[:,:,:,0:1]` |
| 5 | full_normalize | Pass3: (x-mean)*rsqrt(var+eps) + gamma/beta | Full layer_norm | Same as input | N/A |

### Stage 1: data_pipeline
- **Scope**: reader, compute (tilize + untilize only), writer
- **Reference**: `input` (identity passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute does tilize(c_0->c_1) then immediately untilize(c_1->c_17). Single pass, no compute phases.

### Stage 2: reduce_mean
- **Scope**: compute kernel adds reduce_sum_row after tilize
- **Reference**: `x.mean(dim=-1, keepdim=True)` (row-wise mean)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Output shape**: tile-aligned column vector. `output_shape_expr`: `"list(shape[:-1]) + [32]"`
- **Compare slice**: `"[:,:,:,0:1]"` (only column 0 is valid from REDUCE_ROW)
- **Delta from previous**: Adds reduce phase after tilize, writes reduced result via untilize instead of passthrough.

### Stage 3: subtract_mean
- **Scope**: compute adds 2-pass: pass1 reduce_mean, pass2 sub(x, mean)
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Reader now sends row twice. Pass1: tilize+reduce_mean (store in c_24). Pass2: tilize+sub(x,mean)->untilize->write.

### Stage 4: variance_rsqrt
- **Scope**: compute adds pass2 square+reduce+add_eps+rsqrt
- **Reference**: `torch.rsqrt(((x - x.mean(-1,keepdim=True))**2).mean(-1,keepdim=True) + 1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Output shape**: `"list(shape[:-1]) + [32]"`
- **Compare slice**: `"[:,:,:,0:1]"`
- **Delta from previous**: Pass2 now does sub_mean->square->reduce_sum_row->add_eps->rsqrt. Outputs rsqrt_var instead of centered.

### Stage 5: full_normalize
- **Scope**: All 3 passes active; adds pass3 multiply by rsqrt_var + gamma/beta
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma.squeeze(0).squeeze(0).squeeze(0) if gamma is not None else None, bias=beta.squeeze(0).squeeze(0).squeeze(0) if beta is not None else None, eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Reader sends row 3 times. Pass3: tilize, sub(x,mean), mul(centered,rsqrt_var), optionally mul(gamma), add(beta), untilize, write.

### Reader Kernel

Reads RM sticks from DRAM using TensorAccessor. Per tile-row, batches 32 sticks into c_0 (cb_reserve_back Wt, write 32 sticks, cb_push_back Wt). For 3-pass streaming, the same tile-row is read 3 times. At program start, generates reduce scaler (1/W) into c_2 and epsilon tile into c_3 using `dataflow_kernel_lib::prepare_reduce_scaler`. If gamma/beta present, reads their Wt tiles into c_4/c_5 once at startup.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm)`

Uses `fp32_dest_acc_en = true` for precision.

#### Phase 1 (Pass 1): Tilize
```cpp
compute_kernel_lib::tilize<c_0, c_1, InitAndUninit, WaitBlock>(Wt, 1);
```
- In: c_0 [Wt tiles RM data, pushed by reader]
- Out: c_1 [Wt tiles, tilized]

#### Phase 2 (Pass 1): Reduce Mean
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>(
    c_1, c_2, c_24, ReduceInputBlockShape::row(Wt));
```
- A: c_1 [Wt tiles, tilized input, consumed tile-by-tile]
- Scaler: c_2 [1 tile, contains 1/W, persistent]
- Out: c_24 [1 tile, mean col vector, persists for pass2+pass3]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_0 | 0 | freed by tilize |
| c_1 | 0 | freed by reduce (WaitAndPopPerTile) |
| c_24 | 1 | pushed, persists across passes |

#### Phase 3 (Pass 2): Tilize (re-read)
```cpp
compute_kernel_lib::tilize<c_0, c_1, InitAndUninit, WaitBlock>(Wt, 1);
```

#### Phase 4 (Pass 2): Subtract Mean
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_1, c_24, c_25, BinaryInputBlockShape::row(Wt));
```
- A: c_1 [Wt tiles, tilized input, consumed per-tile]
- B: c_24 [1 tile, mean col vector, NoWaitNoPop -- already waited, persists]
- Out: c_25 [Wt tiles, centered data]

#### Phase 5 (Pass 2): Square
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_25, c_25_sq, BinaryInputBlockShape::row(Wt));
```
Note: c_25_sq is a separate CB or c_25 can be reused if square outputs to a different CB. Use c_25 as input, c_1 as output (c_1 is free after phase 4).
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_25, c_1, BinaryInputBlockShape::row(Wt));
```
- A: c_25 [Wt tiles, centered, consumed per-tile]
- Out: c_1 [Wt tiles, centered^2, reusing freed c_1]

#### Phase 6 (Pass 2): Reduce Variance
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>(
    c_1, c_2, c_26, ReduceInputBlockShape::row(Wt));
```
- A: c_1 [Wt tiles, centered^2, consumed per-tile]
- Scaler: c_2 [1/W]
- Out: c_26 [1 tile, variance col vector]

#### Phase 7 (Pass 2): Add Epsilon + Rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_26, c_3, c_27, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_26 [1 tile, var, consumed]
- B: c_3 [1 tile, epsilon, persistent NoWaitNoPop -- but first use so WaitUpfrontNoPop is correct]
- Out: c_27 [1 tile, rsqrt(var+eps), persists for pass3]
- Post-op: rsqrt applied in-DEST before pack

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 | 1 | mean, persists for pass3 |
| c_27 | 1 | rsqrt_var, persists for pass3 |
| c_0, c_1, c_25, c_26 | 0 | freed |

#### Phase 8 (Pass 3): Tilize (re-read)
```cpp
compute_kernel_lib::tilize<c_0, c_1, InitAndUninit, WaitBlock>(Wt, 1);
```

#### Phase 9 (Pass 3): Subtract Mean (again)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_1, c_24, c_25, BinaryInputBlockShape::row(Wt));
```
- A: c_1 [Wt tiles, tilized, consumed per-tile]
- B: c_24 [1 tile, mean, NoWaitNoPop -- still persists from pass1]
- Out: c_25 [Wt tiles, centered]

After this phase, pop c_24 manually: `cb_pop_front(c_24, 1)` (no longer needed).

#### Phase 10 (Pass 3): Multiply by rsqrt(var+eps)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_25, c_27, c_28, BinaryInputBlockShape::row(Wt));
```
- A: c_25 [Wt tiles, centered, consumed per-tile]
- B: c_27 [1 tile, rsqrt_var, NoWaitNoPop -- persists]
- Out: c_28 [Wt tiles, normalized]

After this phase, pop c_27 manually: `cb_pop_front(c_27, 1)`.

#### Phase 11 (Pass 3, conditional): Apply Gamma
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        c_28, c_4, c_16, BinaryInputBlockShape::row(Wt));
}
```
- A: c_28 [Wt tiles, normalized, consumed per-tile]
- B: c_4 [Wt tiles, gamma, NoWaitNoPop -- persistent]
- Out: c_16 [Wt tiles, scaled]

#### Phase 12 (Pass 3, conditional): Apply Beta
```cpp
if constexpr (has_beta) {
    compute_kernel_lib::add<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        c_16, c_5, cb_temp, BinaryInputBlockShape::row(Wt));
}
```
- A: c_16 [Wt tiles, scaled, consumed]
- B: c_5 [Wt tiles, beta, persistent]
- Out: write back to c_16 (or use c_28 as temp and output to c_16)

When neither gamma nor beta: output of phase 10 goes directly to c_16 instead of c_28.

#### Phase 13 (Pass 3): Untilize
```cpp
compute_kernel_lib::untilize<Wt, c_16, c_17, InitAndUninit, WaitBlock>(1);
```
- In: c_16 [Wt tiles, final output in tile format]
- Out: c_17 [Wt tiles, RM sticks for writer]

### Writer Kernel

Follows untilize writer pattern. For each tile-row block: `cb_wait_front(c_17, Wt)`, extract 32 RM sticks at stride `W * 2` bytes, write each stick to DRAM via `TensorAccessor::get_noc_addr(stick_id)` + `noc_async_write`, barrier, `cb_pop_front(c_17, Wt)`.

### Critical Notes

1. **3-pass reader**: Reader must send the same tile-row 3 times per block. Use a `num_passes=3` outer loop around the stick-reading inner loop. Each pass: reserve c_0, read 32 sticks, push c_0. Writer only drains after pass 3.
2. **Persistent scalar CBs**: c_24 (mean) persists from pass1 through pass3; c_27 (rsqrt_var) persists from pass2 through pass3. Manual `cb_pop_front` required after last use.
3. **Gamma/beta as ROW broadcast**: Gamma/beta are (1,1,1,W) tensors. After tilize, they become Wt tiles representing a single row. Use `BroadcastDim::ROW` for these.
4. **Reduce scaler format**: c_2 must be bf16. Use `dataflow_kernel_lib::prepare_reduce_scaler<c_2>(1.0f/W)` or `calculate_and_prepare_reduce_scaler<c_2, SUM, REDUCE_ROW, 32, W>()`.
5. **CB reuse**: c_1 (tilized) is reused for square output in pass2 since tilize data is consumed by sub before square starts.
6. **Gamma/beta without the other**: When only gamma (no beta), phase 11 outputs directly to c_16. When only beta (no gamma), phase 10 outputs to c_16, then phase 12 reads from c_16 and outputs to a temp CB (or c_28) then copies. Simplest: always output phase 10 to c_28, conditionally apply gamma->c_16, conditionally apply beta from c_16->c_28->c_16. The kernel writer should handle the 4 combinations with compile-time flags.

### Implementation Checklist
- [ ] Reader: TensorAccessor RM stick reads, 3-pass per row, generate scaler/eps/gamma/beta
- [ ] Compute: 13 phases using helpers: tilize, reduce(SUM,REDUCE_ROW), sub(COL), square, add(SCALAR)+rsqrt, mul(COL), mul(ROW), add(ROW), untilize
- [ ] Writer: RM stick writes via TensorAccessor, 32 sticks per block
- [ ] CB push/pop balance verified (persistent CBs manually popped)
