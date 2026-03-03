# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), softmax W-small (compute_core)

## Mathematical Definition
```
mean[b,c,h] = sum(input[b,c,h,:]) / W
centered[b,c,h,w] = input[b,c,h,w] - mean[b,c,h]
var[b,c,h] = sum(centered[b,c,h,:]^2) / W
inv_std[b,c,h] = rsqrt(var[b,c,h] + epsilon)
output[b,c,h,w] = gamma[w] * centered[b,c,h,w] * inv_std[b,c,h] + beta[w]
```
Layer normalization across the last (W) dimension of row-major interleaved tensors. Gamma and beta are per-element affine parameters broadcast across all rows.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant for rsqrt |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Layout | ROW_MAJOR | "Input must be row-major" |
| Memory | INTERLEAVED (DRAM) | "Input must be interleaved" |
| Dtype | BFLOAT16 | "Input must be bfloat16" |
| Rank | >= 2 | "Input must be at least 2D" |
| Last dim | multiple of 32 | "W must be tile-aligned" |
| Second-to-last dim | multiple of 32 | "H must be tile-aligned" |

Gamma shape: `(1, 1, 1, W)`, same dtype/layout/memory as input.
Beta shape: `(1, 1, 1, W)`, same dtype/layout/memory as input.

### Output Tensor Specification
- **Shape**: same as input
- **Dtype**: same as input (bfloat16)
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED (DRAM)

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt = 1; reduce produces 1 tile directly |
| epsilon = 0 | Valid but may produce inf for zero-variance rows |
| All-zero row | output = beta (centered = 0, inv_std = rsqrt(eps)) |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize (input_stage) | Read RM sticks, tilize in compute | Add gamma/beta reading (single-row RM tensors, tilized in compute) |
| Compute | softmax (compute_core) | Multi-phase row reduction with CB persistence | Replace max/exp/recip phases with mean/square/rsqrt/affine phases |
| Writer | untilize (output_stage) | Untilize in compute, write RM sticks | Standard RM stick writer |

### Work Distribution
- **Work unit**: Row of tiles (Wt tiles along W dimension, one tile-row of height 32)
- **Grid**: Single core (initial version)
- **Work per core**: `num_rows = N_outer * Ht` where `N_outer = volume / (H * W)`, `Ht = H / 32`
- **Remainder**: N/A (single core)

### Data Flow
Reader reads 32 RM sticks per block from input (and once from gamma/beta), pushes to input CBs. Compute tilizes input, then performs 7 phases of tile computation (mean reduce, subtract mean, square, variance reduce+rsqrt, multiply inv_std, multiply gamma, add beta), then untilizes result. Writer extracts 32 RM sticks per block and writes to DRAM.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input_rm | Input RM sticks (tilize input) | Reader | Compute (tilize) | Wt | Block |
| c_1 | cb_gamma_rm | Gamma RM sticks (tilize input) | Reader | Compute (tilize) | Wt | Program (loaded once) |
| c_2 | cb_beta_rm | Beta RM sticks (tilize input) | Reader | Compute (tilize) | Wt | Program (loaded once) |
| c_8 | cb_scaler | Reduce scaler (1/W packed bf16) | Reader | Compute (reduce) | 1 | Program (loaded once, never popped) |
| c_9 | cb_eps | Epsilon scaler tile | Reader | Compute (add eps) | 1 | Program (loaded once, never popped) |
| c_16 | cb_out_rm | Output RM sticks (untilize output) | Compute (untilize) | Writer | Wt | Block |
| c_24 | cb_input_tiled | Tilized input (full row) | Compute (tilize) | Compute (reduce, sub) | Wt | Row (persists Phase 1-2) |
| c_25 | cb_mean | Mean tile (REDUCE_ROW output) | Compute (reduce) | Compute (sub) | 1 | Phase (Phase 1-2) |
| c_26 | cb_centered | x - mean intermediate | Compute (sub) | Compute (square, mul inv_std) | Wt | Row (persists Phase 2-5) |
| c_27 | cb_var_sq | centered^2 intermediate | Compute (square) | Compute (reduce var) | Wt | Phase (Phase 3-4) |
| c_28 | cb_inv_std | inv_std tile (rsqrt output) | Compute (reduce+rsqrt) | Compute (mul) | 1 | Phase (Phase 4-5) |
| c_29 | cb_gamma_tiled | Tilized gamma (full row) | Compute (tilize) | Compute (mul gamma) | Wt | Program (persists all rows) |
| c_30 | cb_beta_tiled | Tilized beta (full row) | Compute (tilize) | Compute (add beta) | Wt | Program (persists all rows) |
| c_31 | cb_normed | x_centered * inv_std intermediate | Compute (mul) | Compute (mul gamma) | Wt | Phase (Phase 5-6) |

All CB page sizes are tile_size (e.g., 2048 bytes for bf16 32x32 tiles). The RM input CBs (c_0, c_1, c_2) also use tile-sized pages in symmetric tilize mode (32 sticks x stick_width = Wt x tile_size).

### Kernel Arguments

**Compile-time** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | Bytes per RM stick (W * 2 for bf16) |
| Reader | 1+ | TensorAccessorArgs(input) | varies | Input buffer accessor |
| Reader | N+0 | gamma_stick_size | uint32_t | Bytes per gamma stick (same as stick_size) |
| Reader | N+1+ | TensorAccessorArgs(gamma) | varies | Gamma buffer accessor |
| Reader | M+0 | beta_stick_size | uint32_t | Bytes per beta stick (same as stick_size) |
| Reader | M+1+ | TensorAccessorArgs(beta) | varies | Beta buffer accessor |
| Compute | 0 | num_rows | uint32_t | Total tile-rows to process |
| Compute | 1 | Wt | uint32_t | Tiles per row (W / 32) |
| Writer | 0 | stick_size | uint32_t | Bytes per output RM stick |
| Writer | 1+ | TensorAccessorArgs(output) | varies | Output buffer accessor |

**Runtime** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer DRAM address |
| Reader | 1 | gamma_addr | uint32_t | Gamma buffer DRAM address |
| Reader | 2 | beta_addr | uint32_t | Beta buffer DRAM address |
| Reader | 3 | num_rows | uint32_t | Total tile-rows to process |
| Reader | 4 | Wt | uint32_t | Tiles per row |
| Reader | 5 | start_stick_id | uint32_t | First stick ID for this core |
| Reader | 6 | scaler_value | uint32_t | 1/W as packed bf16 bits |
| Reader | 7 | eps_value | uint32_t | epsilon as packed bf16 bits |
| Writer | 0 | dst_addr | uint32_t | Output buffer DRAM address |
| Writer | 1 | num_rows | uint32_t | Total tile-rows to process |
| Writer | 2 | Wt | uint32_t | Tiles per row |
| Writer | 3 | start_stick_id | uint32_t | First output stick ID |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB (c_8) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) / 4 tiles (f32 half-sync)
- [x] RM CBs count pages in tiles (symmetric tilize mode); tile CBs count in tiles

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` (see tolerances per stage below)
- Test shapes:

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration in W | `(1, 1, 32, 128)` |
| Multi-tile HW | Multi-row + multi-col | `(1, 1, 64, 128)` |
| Non-square | W != H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (cb_input_rm) | Wt | RM→Tile (symmetric tilize) | All | Block (per tile-row) |
| c_1 (cb_gamma_rm) | Wt | RM→Tile (symmetric tilize) | Row0 (single row) | Program |
| c_2 (cb_beta_rm) | Wt | RM→Tile (symmetric tilize) | Row0 (single row) | Program |
| c_8 (cb_scaler) | 1 | Tile | Row0 (scaler) | Program (never popped) |
| c_9 (cb_eps) | 1 | Tile | Row0 (scaler) | Program (never popped) |
| c_16 (cb_out_rm) | Wt | Tile→RM (untilize output) | All | Block |
| c_24 (cb_input_tiled) | Wt | Tile | All | Row |
| c_25 (cb_mean) | 1 | Tile | Col0 (REDUCE_ROW output) | Phase |
| c_26 (cb_centered) | Wt | Tile | All | Row |
| c_27 (cb_var_sq) | Wt | Tile | All | Phase |
| c_28 (cb_inv_std) | 1 | Tile | Col0 (REDUCE_ROW output) | Phase |
| c_29 (cb_gamma_tiled) | Wt | Tile | Row0 (broadcast ROW) | Program |
| c_30 (cb_beta_tiled) | Wt | Tile | Row0 (broadcast ROW) | Program |
| c_31 (cb_normed) | Wt | Tile | All | Phase |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 2 (sub mean) | SUB | All | Col0 | COL |
| 3 (square) | SQUARE | All | (self) | NONE |
| 5 (mul inv_std) | MUL | All | Col0 | COL |
| 6 (mul gamma) | MUL | All | Row0 | ROW |
| 7 (add beta) | ADD | All | Row0 | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader (tilize) + writer (untilize), compute = tilize then untilize (identity) | `input.clone()` |
| 2 | subtract_mean | Phase 1 (reduce mean) + Phase 2 (subtract mean) | `x - x.mean(dim=-1, keepdim=True)` |
| 3 | full_layer_norm | Phases 3-7 (square, variance, inv_std, gamma, beta) | `F.layer_norm(x, [W], gamma, beta, eps)` |

### Stage 1: data_pipeline
- **Scope**: reader kernel (read input sticks, gamma sticks, beta sticks), compute kernel (tilize c_0 to c_24, untilize c_24 to c_16), writer kernel (write output sticks)
- **Reference**: `return input_tensor.clone()`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes input then immediately untilizes (identity pass-through). Gamma/beta read but unused. Phases 1-7 skipped.

### Stage 2: subtract_mean
- **Scope**: compute kernel adds Phase 1 (reduce SUM with scaler 1/W) and Phase 2 (sub broadcast COL)
- **Reference**: `return input_tensor - input_tensor.to(torch.float32).mean(dim=-1, keepdim=True).to(torch.bfloat16)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Compute kernel now performs tilize, reduce mean, subtract mean, untilize (instead of tilize + untilize)

### Stage 3: full_layer_norm
- **Scope**: compute kernel adds Phases 3-7 (square, variance reduce+rsqrt, mul inv_std, mul gamma, add beta)
- **Reference**: `return torch.nn.functional.layer_norm(input_tensor.to(torch.float32), [input_tensor.shape[-1]], weight=gamma.flatten().to(torch.float32), bias=beta.flatten().to(torch.float32), eps=1e-5).to(torch.bfloat16)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: All 7 compute phases active. Gamma and beta tilized and applied.

### Reader Kernel

Reads 3 tensors from DRAM:
1. **Gamma/beta** (once at start): Read 32 RM sticks (only first stick has data, rest are padding for tile alignment) into c_1 and c_2 respectively, push Wt pages each. Uses tilize-style batching with TensorAccessor.
2. **Scaler** (once at start): Generate reduce scaler tile (1/W) in c_8 using `dataflow_kernel_lib::prepare_reduce_scaler<c_8>(1.0f / W)`.
3. **Epsilon** (once at start): Generate epsilon scaler tile in c_9 using `dataflow_kernel_lib::prepare_reduce_scaler<c_9>(epsilon)`.
4. **Input** (per row): Read 32 sticks into c_0 using tilize batch pattern (reserve Wt, read 32 sticks, push Wt). Uses TensorAccessor with stick_id.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_8, c_16)` — three-argument form since srcA (c_0) and srcB (c_8) differ from output (c_16).

**One-time setup**: Tilize gamma from c_1 to c_29, tilize beta from c_2 to c_30.

```cpp
compute_kernel_lib::tilize<c_1, c_29, InitUninitMode::InitOnly>(Wt, 1);
compute_kernel_lib::tilize<c_2, c_30, InitUninitMode::UninitOnly>(Wt, 1);
```

**Per-row loop** (for each of num_rows tile-rows):

#### Phase 0: Tilize input
```cpp
compute_kernel_lib::tilize<c_0, c_24>(Wt, 1);
```
- In: c_0 [Wt pages, from reader]
- Out: c_24 [Wt tiles, tiled format]

#### Phase 1: Reduce mean
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>(
    c_24, c_8, c_25,
    ReduceInputBlockShape::row(Wt));
```
- A: c_24 [Wt tiles, FRESHLY PUSHED by Phase 0, kept via WaitUpfrontNoPop]
- Scaler: c_8 [1 tile, 1/W, never popped]
- Out: c_25 [1 tile, contains row means in Col0]

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 | Wt | waited, not popped — persists for Phase 2 |
| c_25 | 1 | freshly pushed |

#### Phase 2: Subtract mean
```cpp
compute_kernel_lib::sub<
    BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,    // c_24 already waited from Phase 1
    BinaryInputPolicy::WaitUpfrontPopAtEnd,  // c_25 waited + popped after
    BinaryOutputPolicy::Bulk>(
    c_24, c_25, c_26,
    BinaryInputBlockShape::row(Wt));
// Manual pop of c_24 after sub
cb_pop_front(c_24, Wt);
```
- A: c_24 [Wt tiles, ALREADY WAITED from Phase 1, manually popped after]
- B: c_25 [1 tile, waited and popped at end by helper]
- Out: c_26 [Wt tiles, bulk pushed]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 | 0 | freed (manually popped) |
| c_25 | 0 | freed (popped at end by helper) |
| c_26 | Wt | freshly pushed — persists for Phase 3 and Phase 5 |

#### Phase 3: Square centered values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,  // c_26 persists for Phase 5
    BinaryOutputPolicy::Bulk>(
    c_26, c_27,
    BinaryInputBlockShape::row(Wt));
```
- A: c_26 [Wt tiles, FRESHLY PUSHED by Phase 2, kept via WaitUpfrontNoPop]
- Out: c_27 [Wt tiles, bulk pushed]

#### Phase 4: Reduce variance + add epsilon + rsqrt
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontPopAtEnd>(
    c_27, c_8, c_28,
    ReduceInputBlockShape::row(Wt),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst_idx) {
        // Add epsilon: load eps tile, add to DST
        // Then compute rsqrt
        add_tiles_bcast<BroadcastType::SCALAR>(c_9, dst_idx);  // var + eps
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```

NOTE: The post-reduce lambda for epsilon addition requires manual tile operations (add eps scalar tile from c_9, then rsqrt). The exact implementation may need to use `copy_tile` to load eps into DST and manual add, since `add_tiles_bcast` is a binary unpack+math op that conflicts with reduce's DST state. The kernel writer should implement this as: (1) reduce SUM to get variance in DST, (2) pack variance to a temp CB, (3) binary add eps via `add<SCALAR>` helper, (4) apply rsqrt via post-op on the add. Alternatively, the kernel writer can split this into separate reduce + binary_add_eps + rsqrt steps using intermediate CBs. The key requirement is: `inv_std = rsqrt(var + eps)` ends up in c_28 as 1 tile.

- A: c_27 [Wt tiles, freshly pushed by Phase 3, popped at end]
- Scaler: c_8 [1 tile, 1/W, never popped]
- Out: c_28 [1 tile, contains inv_std in Col0]

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| c_27 | 0 | freed (popped at end) |
| c_28 | 1 | freshly pushed |
| c_26 | Wt | still waited from Phase 3 — persists for Phase 5 |

#### Phase 5: Multiply by inv_std
```cpp
compute_kernel_lib::mul<
    BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,           // c_26 already waited
    BinaryInputPolicy::WaitUpfrontPopAtEnd,   // c_28 waited + popped
    BinaryOutputPolicy::Bulk>(
    c_26, c_28, c_31,
    BinaryInputBlockShape::row(Wt));
cb_pop_front(c_26, Wt);  // Manual pop after use
```
- A: c_26 [Wt tiles, ALREADY WAITED from Phase 3, manually popped after]
- B: c_28 [1 tile, waited and popped at end by helper]
- Out: c_31 [Wt tiles, bulk pushed]

**CB state after Phase 5:**
| CB | Tiles | State |
|----|-------|-------|
| c_26 | 0 | freed (manually popped) |
| c_28 | 0 | freed (popped at end by helper) |
| c_31 | Wt | freshly pushed |

#### Phase 6: Multiply by gamma
```cpp
compute_kernel_lib::mul<
    BroadcastDim::ROW,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,   // c_31 consumed
    BinaryInputPolicy::NoWaitNoPop,           // c_29 persists (program lifetime)
    BinaryOutputPolicy::Bulk>(
    c_31, c_29, c_26,                         // Reuse c_26 as output
    BinaryInputBlockShape::row(Wt));
```
- A: c_31 [Wt tiles, freshly pushed, consumed and popped]
- B: c_29 [Wt tiles, gamma, ALREADY WAITED at program start, never popped]
- Out: c_26 [Wt tiles, reused CB, bulk pushed]

#### Phase 7: Add beta
```cpp
compute_kernel_lib::add<
    BroadcastDim::ROW,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,   // c_26 consumed
    BinaryInputPolicy::NoWaitNoPop,           // c_30 persists
    BinaryOutputPolicy::Bulk>(
    c_26, c_30, c_24,                         // Reuse c_24 as output
    BinaryInputBlockShape::row(Wt));
```
- A: c_26 [Wt tiles, freshly pushed, consumed and popped]
- B: c_30 [Wt tiles, beta, never popped]
- Out: c_24 [Wt tiles, reused CB, bulk pushed]

#### Phase 8: Untilize output
```cpp
compute_kernel_lib::untilize<Wt, c_24, c_16,
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::WaitBlock>(1);
```
- In: c_24 [Wt tiles, freshly pushed by Phase 7]
- Out: c_16 [Wt tiles, RM format for writer]

### Writer Kernel

Standard RM stick writer following untilize output_stage pattern:
1. Wait for Wt tiles in c_16 per block.
2. Get L1 read pointer, iterate 32 sticks.
3. For each stick: compute output page_id = `block_idx * 32 + stick_idx`, write via TensorAccessor + `noc_async_write`.
4. Barrier per block, pop c_16.

### Critical Notes
1. **Gamma/beta CB persistence**: c_29 and c_30 are loaded once and never popped. The binary_op helper's `NoWaitNoPop` policy for input_b ensures they persist across all rows. The compute kernel must call `cb_wait_front(c_29, Wt)` and `cb_wait_front(c_30, Wt)` once before the row loop.
2. **Scaler/epsilon persistence**: c_8 and c_9 are never popped. The reduce helper accesses them with indexed reads. The compute kernel must `cb_wait_front(c_8, 1)` and `cb_wait_front(c_9, 1)` once before the row loop.
3. **Phase 4 epsilon addition**: The post-reduce lambda in Phase 4 is architecturally complex. The kernel writer may need to split Phase 4 into: (a) reduce SUM to temp CB, (b) add epsilon via binary add SCALAR, (c) apply rsqrt. This adds an intermediate CB but avoids DST conflicts. The CB c_27 can be reused as the temp CB since it is freed at Phase 4.
4. **CB reuse**: c_24 and c_26 are reused across phases (input tiled, then output of add beta / mul gamma respectively). This is safe because their previous contents are fully consumed (popped) before reuse.
5. **Tilize init/uninit**: The one-time gamma/beta tilize uses InitOnly/UninitOnly to bracket the two calls. The per-row tilize of input uses InitAndUninit (standalone). The untilize similarly uses InitAndUninit. If register reconfiguration is needed between tilize and the binary/reduce phases, use `ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure` on the first binary/reduce call.

### Implementation Checklist
- [ ] Reader: 3 TensorAccessors (input, gamma, beta), generate scaler (1/W) and eps tiles, tilize-batch pattern for all 3 inputs
- [ ] Compute: 9 phases (tilize, reduce mean, sub mean, square, reduce var+rsqrt, mul inv_std, mul gamma, add beta, untilize) using helpers: tilize, reduce, sub(COL), square, mul(COL), mul(ROW), add(ROW), untilize
- [ ] Writer: TensorAccessor for output sticks, 32-stick-per-block drain pattern, barrier per block
- [ ] CB push/pop balance verified per row iteration
