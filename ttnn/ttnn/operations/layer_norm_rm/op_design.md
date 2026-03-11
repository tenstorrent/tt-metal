# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), batch_norm (compute_core)

## Mathematical Definition
```
For each row of width W:
  mean = sum(x_i) / W
  centered = x_i - mean
  var = sum(centered^2) / W
  inv_std = rsqrt(var + eps)
  x_norm = centered * inv_std
  output = x_norm * gamma + beta   (if gamma/beta provided)
```
Row-wise normalization across the last dimension (W). Each row of 32 tiles is processed as one block.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Stability constant for variance |
| gamma | Tensor | No | shape (1,1,1,W), bf16, RM | None | Per-element scale |
| beta | Tensor | No | shape (1,1,1,W), bf16, RM | None | Per-element shift |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | BFLOAT16 | Non-bf16 input not supported |
| layout | ROW_MAJOR | Tile layout input not supported |
| memory | Interleaved DRAM | Sharded not supported |
| shape | H,W multiples of 32 | Input must be tile-aligned |

### Output Tensor Specification
- **Shape**: same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: Interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce_row produces 1 tile per row |
| gamma provided, beta None | Apply scale only, skip bias add |
| gamma None, beta None | Output pure normalization |
| Large batch (N*C*H >> grid) | Cliff core handles remainder blocks |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize analysis | input_stage | Add scaler/eps fill, optional gamma/beta reads |
| Compute | batch_norm analysis + NEW | compute_core | Replace channel-broadcast with row-reduce; use helpers |
| Writer | untilize analysis | output_stage | Unchanged pattern: extract 32 RM sticks per block |

### Work Distribution
- **Work unit**: tile-row block (32 sticks x full width = Wt tiles)
- **Grid**: 1D linear, up to grid_size.x * grid_size.y cores
- **Work per core**: nblocks_per_core = ceil(total_blocks / num_cores)
- **Remainder**: cliff core gets nblocks % nblocks_per_core blocks
- **total_blocks**: N * C * H / 32 (total tile-rows)

### Data Flow
Reader batches 32 RM sticks into c_0 per block. Compute tilizes, normalizes row-by-row through 7-10 phases, untilizes result. Writer extracts 32 RM sticks from c_16 and writes to DRAM. Gamma/beta (if present) are read once at program start, tilized, and persist in dedicated CBs.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_rm_input | RM sticks from DRAM | Reader | Compute(tilize) | Wt | Block |
| c_1 | cb_gamma | Tilized gamma (optional) | Compute(tilize) | Compute(mul) | Wt | Program |
| c_2 | cb_beta | Tilized beta (optional) | Compute(tilize) | Compute(add) | Wt | Program |
| c_8 | cb_scaler | Reduce scaler (1/W) | Reader | Compute(reduce) | 1 | Program |
| c_9 | cb_eps | Epsilon constant tile | Reader | Compute(add) | 1 | Program |
| c_16 | cb_rm_output | Untilized RM output | Compute(untilize) | Writer | Wt | Block |
| c_24 | cb_tilized | Tilized input / reused intermediate | Compute | Compute | Wt | Block |
| c_25 | cb_mean | Row mean / reused for var | Compute(reduce) | Compute(sub) | 1 | Block |
| c_26 | cb_centered | x - mean (persists for normalize) | Compute(sub) | Compute(square,mul) | Wt | Block |
| c_27 | cb_squared | (x-mean)^2 / reused for affine output | Compute(square) | Compute(reduce) | Wt | Block |
| c_28 | cb_inv_std | rsqrt(var + eps) | Compute(add+rsqrt) | Compute(mul) | 1 | Block |

### Kernel Arguments

**Compile-time** (Reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * 2 bytes (bf16) |
| 1+ | input_accessor_args | TensorAccessorArgs | Input buffer metadata |
| +1 | has_gamma | uint32_t | 1 if gamma present |
| +2 | has_beta | uint32_t | 1 if beta present |
| +3 | gamma_accessor_args | TensorAccessorArgs | Gamma buffer metadata (if present) |
| +4 | beta_accessor_args | TensorAccessorArgs | Beta buffer metadata (if present) |

**Runtime** (Reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | num_blocks | uint32_t | Tile-row blocks for this core |
| 2 | start_stick_id | uint32_t | Global stick index to start reading |
| 3 | Wt | uint32_t | Tiles per row (W/32) |
| 4 | W | uint32_t | Width in elements |
| 5 | gamma_addr | uint32_t | Gamma buffer address (0 if absent) |
| 6 | beta_addr | uint32_t | Beta buffer address (0 if absent) |

**Compile-time** (Compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row |
| 1 | num_blocks | uint32_t | Blocks for this core |
| 2 | has_gamma | uint32_t | 1 if gamma present |
| 3 | has_beta | uint32_t | 1 if beta present |

**Compile-time** (Writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * 2 bytes (bf16) |
| 1+ | output_accessor_args | TensorAccessorArgs | Output buffer metadata |

**Runtime** (Writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_blocks | uint32_t | Tile-row blocks for this core |
| 2 | start_stick_id | uint32_t | Global stick index to start writing |
| 3 | Wt | uint32_t | Tiles per row |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count per occurrence
- [x] Reduce scaler CB (c_8) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) / 4 tiles (f32)
- [x] RM CBs count pages in tiles (stick data fills tile-sized pages)
- [x] Wt compile-time arg used for untilize block_width_tiles template param

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs PyTorch `F.layer_norm` reference

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
| c_0 | Wt | tile-sized (RM data) | All | Block (reused for gamma/beta setup) |
| c_1 | Wt | tile | Row0 (gamma broadcast) | Program |
| c_2 | Wt | tile | Row0 (beta broadcast) | Program |
| c_8 | 1 | tile (bf16) | Row0 (scaler) | Program |
| c_9 | 1 | tile (bf16) | Scalar [0,0] | Program |
| c_16 | Wt | tile-sized (RM data) | All | Block |
| c_24 | Wt | tile | All | Block (reused) |
| c_25 | 1 | tile | Col0 (reduce_row output) | Block (reused) |
| c_26 | Wt | tile | All | Block |
| c_27 | Wt | tile | All | Block (reused) |
| c_28 | 1 | tile | Col0 | Block |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 3 (sub mean) | SUB | c_24: All | c_25: Col0 (reduce_row output) | COL |
| 7 (mul inv_std) | MUL | c_26: All | c_28: Col0 | COL |
| 8 (mul gamma) | MUL | c_24: All | c_1: Row0 (tilized 1-row param) | ROW |
| 9 (add beta) | ADD | c_27 or c_24: All | c_2: Row0 (tilized 1-row param) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | data_pipeline | tilize + untilize | identity (passthrough) | Same as input | N/A |
| 2 | center_and_square | reduce_mean + sub + square | (x - mean(x))^2 | Same as input | N/A |
| 3 | normalize | reduce_var + add_eps_rsqrt + mul | normalized x (no affine) | Same as input | N/A |
| 4 | affine | gamma_tilize + beta_tilize + mul + add | full layer_norm | Same as input | N/A |

### Stage 1: data_pipeline
- **Scope**: reader, compute (tilize + untilize), writer
- **Reference**: `input.clone()`
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.01, atol=0.01
- **Phases**: tilize(c_0 -> c_24), untilize(c_24 -> c_16)
- **CB bypass**: Compute tilizes input, immediately untilizes to output. No normalization.

### Stage 2: center_and_square
- **Scope**: compute kernel adds phases 2-4
- **Reference**: `(input - input.mean(dim=-1, keepdim=True)) ** 2`
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Reader fills scaler CB (1/W). Compute adds reduce_row(mean), sub(COL), square between tilize and untilize.

### Stage 3: normalize
- **Scope**: compute kernel adds phases 5-7
- **Reference**: `mean = input.mean(-1, keepdim=True); centered = input - mean; var = centered.pow(2).mean(-1, keepdim=True); (centered) / torch.sqrt(var + 1e-5)`
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Reader fills eps CB. Compute adds reduce_row(var), add_eps+rsqrt, mul_normalize. Output changes from squared to normalized.

### Stage 4: affine
- **Scope**: reader adds gamma/beta reads, compute adds tilize(gamma/beta) + mul_gamma + add_beta
- **Reference**: `torch.nn.functional.layer_norm(input, [input.shape[-1]], weight=gamma.squeeze(0).squeeze(0).squeeze(0), bias=beta.squeeze(0).squeeze(0).squeeze(0), eps=1e-5)`
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Reader reads gamma/beta sticks, pushes to c_0. Compute tilizes gamma->c_1, beta->c_2 once at start. Main loop adds mul(ROW) + add(ROW) after normalize.

### Reader Kernel
Reads RM sticks from DRAM using TensorAccessor. Per block: reserve c_0 (Wt pages), read 32 sticks via noc_async_read, barrier, push Wt pages. At program start: fill c_8 with reduce scaler (1/W) via `prepare_reduce_scaler`, fill c_9 with eps via `fill_with_val`. If gamma/beta present: read 32 sticks each into c_0, push to compute for tilize.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_24, c_8, c_16);`

**Pre-loop setup (once):**
- Wait c_8 (scaler, 1 tile) — persists entire program
- Wait c_9 (eps, 1 tile) — persists entire program
- If has_gamma: tilize c_0 -> c_1 (Wt tiles, persists)
- If has_beta: tilize c_0 -> c_2 (Wt tiles, persists)

**Main loop (per block):**

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<c_0, c_24,
    tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);
```
- In: c_0 [Wt pages, waited+popped by helper]
- Out: c_24 [Wt tiles pushed]

#### Phase 2: Reduce Row (mean)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop>(
    c_24, c_8, c_25,
    ReduceInputBlockShape::of(1, Wt));
```
- A: c_24 [Wt tiles, FRESHLY PUSHED by Phase 1, WaitUpfrontNoPop — tiles persist]
- Scaler: c_8 [1 tile, already waited at startup, persistent]
- Out: c_25 [1 tile pushed, col0 = mean]

#### Phase 3: Sub mean (centering)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::NoWaitNoPop>(
    c_24, c_25, c_26,
    BinaryInputBlockShape::of(1, Wt));
cb_pop_front(c_24, Wt);
cb_pop_front(c_25, 1);
```
- A: c_24 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop]
- B: c_25 [1 tile, FRESHLY PUSHED by Phase 2, NoWaitNoPop — must be waited by binary helper? No, NoWaitNoPop means caller manages. Need explicit wait before this call.]
- Out: c_26 [Wt tiles pushed]

**CORRECTION**: c_25 was just pushed by reduce. With NoWaitNoPop on B, the binary_op helper will NOT wait for c_25. We need an explicit `cb_wait_front(c_25, 1)` before calling sub. Or use WaitUpfrontNoPop for B policy.

Revised Phase 3:
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_24, c_25, c_26,
    BinaryInputBlockShape::of(1, Wt));
cb_pop_front(c_24, Wt);
```
- A: c_24 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop — caller pops]
- B: c_25 [1 tile, WaitUpfrontPopAtEnd — helper waits and pops at end]
- Out: c_26 [Wt tiles pushed]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 | 0 | freed (manually popped) |
| c_25 | 0 | freed (popped by helper) |
| c_26 | Wt | freshly pushed, needed for Phase 4 and Phase 7 |

#### Phase 4: Square centered
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_26, c_27,
    BinaryInputBlockShape::of(1, Wt));
```
- A: c_26 [Wt tiles, FRESHLY PUSHED by Phase 3, WaitUpfrontNoPop — tiles persist for Phase 7]
- Out: c_27 [Wt tiles pushed]

#### Phase 5: Reduce Row (variance)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    c_27, c_8, c_25,
    ReduceInputBlockShape::of(1, Wt));
```
- A: c_27 [Wt tiles, FRESHLY PUSHED by Phase 4, default WaitAndPopPerTile]
- Scaler: c_8 [1 tile, persistent]
- Out: c_25 [1 tile pushed, col0 = variance]. CB reused from mean (was freed).

#### Phase 6: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_25, c_9, c_28,
    BinaryInputBlockShape::of(1, 1),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_25 [1 tile, FRESHLY PUSHED by Phase 5, WaitAndPopPerTile — waited and popped]
- B: c_9 [1 tile, persistent from startup, NoWaitNoPop]
- Out: c_28 [1 tile pushed = rsqrt(var + eps)]
- Post-op: rsqrt applied in DEST before pack

#### Phase 7: Multiply by inv_std (normalize)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_26, c_28, c_24,
    BinaryInputBlockShape::of(1, Wt));
cb_pop_front(c_26, Wt);
```
- A: c_26 [Wt tiles, ALREADY WAITED from Phase 4, NoWaitNoPop — caller pops]
- B: c_28 [1 tile, FRESHLY PUSHED by Phase 6, WaitUpfrontPopAtEnd — helper waits+pops]
- Out: c_24 [Wt tiles pushed = normalized]. CB reused.

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_26 | 0 | freed (manually popped) |
| c_28 | 0 | freed (popped by helper) |
| c_24 | Wt | freshly pushed — normalized output |

#### Phase 8 (optional): Multiply gamma
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        c_24, c_1, c_27,
        BinaryInputBlockShape::of(1, Wt));
}
```
- A: c_24 [Wt tiles, WaitAndPopPerTile]
- B: c_1 [Wt tiles, persistent gamma, NoWaitNoPop]
- Out: c_27 [Wt tiles pushed]

#### Phase 9 (optional): Add beta
```cpp
if constexpr (has_beta) {
    uint32_t affine_in = has_gamma ? c_27 : c_24;
    compute_kernel_lib::add<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        affine_in, c_2, cb_pre_untilize,
        BinaryInputBlockShape::of(1, Wt));
}
```
- A: result from Phase 8 or Phase 7 [Wt tiles, WaitAndPopPerTile]
- B: c_2 [Wt tiles, persistent beta, NoWaitNoPop]
- Out: cb_pre_untilize [Wt tiles pushed]

**CB routing for untilize input:**
- No gamma, no beta: untilize from c_24
- Gamma only: untilize from c_27
- Beta only: untilize from c_27 (normalized -> c_24, add_beta c_24 -> c_27)
- Gamma + beta: untilize from c_24 (gamma c_24->c_27, beta c_27->c_24)

Compute kernel determines cb_pre_untilize at compile time based on has_gamma/has_beta:
```
cb_pre_untilize = (!has_gamma && !has_beta) ? c_24
                : (has_gamma && has_beta)   ? c_24
                : c_27;
```
When beta only (no gamma): sub output in c_24, add_beta writes c_24->c_27, untilize from c_27.
When gamma + beta: gamma writes c_24->c_27, beta writes c_27->c_24, untilize from c_24.

#### Phase 10: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_pre_untilize, c_16,
    untilize_config::InitUninitMode::InitAndUninit>(1);
```
- In: cb_pre_untilize [Wt tiles, freshly pushed by last compute phase]
- Out: c_16 [Wt tile-pages of RM data pushed]

**End of main loop (per block).** Pop c_8 and c_9 after all blocks.
If has_gamma: pop c_1 (Wt). If has_beta: pop c_2 (Wt).

### Writer Kernel
Waits on c_16 (Wt pages per block). For each block: `cb_wait_front(c_16, Wt)`, get base L1 addr via `get_read_ptr(c_16)`, iterate 32 rows writing one RM stick per row via `noc_async_write` to TensorAccessor address. Barrier per block, then `cb_pop_front(c_16, Wt)`.

### Critical Notes
1. **Scaler CB format**: c_8 must be bfloat16. Use `prepare_reduce_scaler<c_8>(1.0f / W)` in reader.
2. **Eps fill**: c_9 filled via `fill_with_val` pattern (bfloat16 packed scalar). Use `FILL_WITH_VALUE` with double-packed bf16.
3. **NoWaitNoPop manual pops**: After Phase 3 and Phase 7, caller must manually `cb_pop_front` the A operand since NoWaitNoPop does not pop.
4. **Gamma/beta tilize reuses c_0**: Reader sequentially pushes gamma sticks, then beta sticks, then main loop sticks all through c_0. Compute tilizes each in order.
5. **Untilize block_width_tiles is compile-time**: Wt must be a compile-time arg for the untilize template.
6. **CB c_25 reused**: Mean (Phase 2) and variance (Phase 5) share c_25 since mean is consumed before variance is produced.

### Implementation Checklist
- [ ] Reader: TensorAccessor RM reads, scaler/eps fill, optional gamma/beta reads
- [ ] Compute: 10 phases using helpers: tilize, reduce(x2), sub(COL), square, add(SCALAR)+rsqrt, mul(COL), mul(ROW), add(ROW), untilize
- [ ] Writer: TensorAccessor RM writes, 32 sticks per block extraction
- [ ] CB push/pop balance verified per block
