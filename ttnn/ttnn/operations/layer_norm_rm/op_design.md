# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), batch_norm (compute_core)

## Mathematical Definition
```
mean[b,h]      = sum(x[b,h,:]) / W
centered[b,h,w]= x[b,h,w] - mean[b,h]
var[b,h]       = sum(centered[b,h,:]^2) / W
output[b,h,w]  = centered[b,h,w] * rsqrt(var[b,h] + eps) [* gamma[w] + beta[w]]
```
Per-row normalization: each row of width W is independently normalized. Optional affine transform with gamma/beta broadcast along rows.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_tensor | Tensor | Yes | 4D, H%32==0, W%32==0 | - | RM interleaved bfloat16 |
| gamma | Tensor | No | shape (1,1,1,W), bfloat16 | None | Per-element scale |
| beta | Tensor | No | shape (1,1,1,W), bfloat16 | None | Per-element bias |
| epsilon | float | No | >0 | 1e-5 | Numerical stability constant |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Layout | ROW_MAJOR |
| Memory | Interleaved DRAM |
| Dtype | BFLOAT16 |
| Shape | (N, C, H, W), H and W multiples of 32 |

### Output Tensor Specification
- **Shape**: Same as input (N, C, H, W)
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: Interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| gamma=None, beta=None | Skip affine phases |
| gamma only (no beta) | Apply scale only |
| beta only (no gamma) | Apply bias only |
| W=32 (single tile width) | Wt=1, reduce is trivial |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize_analysis | input_stage | Add gamma/beta/eps reads, reduce scaler prep |
| Compute | batch_norm_analysis + new | compute_core | Replace pre-computed stats with inline reduce; add tilize/untilize phases |
| Writer | untilize_analysis | output_stage | Reuse RM stick writer pattern |

### Work Distribution
- **Work unit**: Block = 1 tile-row (32 sticks, Wt tiles wide). Each block is one independent row normalization.
- **Grid**: 1D linearized, up to full compute grid
- **Work per core**: `ceil(num_blocks / num_cores)` blocks; cliff core gets remainder
- **Remainder**: Last core processes fewer blocks via `split_blocks_for_tilize`

### Data Flow
RM sticks read from DRAM -> tilize in compute -> reduce_row for mean -> subtract mean -> square -> reduce_row for variance -> add epsilon + rsqrt -> multiply inv_std -> optional gamma/beta affine -> untilize in compute -> RM sticks written to DRAM.

All compute operates per-block (one tile-row = Wt tiles). Mean and variance are [1,1] tiles per row (REDUCE_ROW output). Centered tiles are kept in a persistent CB for reuse in both variance and normalize phases.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_rm_in | RM sticks from reader (tilize input) | Reader | Compute | Wt | Block |
| c_1 | cb_tilized | Tilized tiles | Compute | Compute | Wt | Block |
| c_2 | cb_reduce_scaler | Reduce scaler (1/W bfloat16) | Reader | Compute | 1 | Program |
| c_3 | cb_eps | Epsilon constant tile | Reader | Compute | 1 | Program |
| c_4 | cb_mean | Row mean [Ht=1,1] | Compute | Compute | 1 | Block |
| c_5 | cb_centered | x - mean (persistent for reuse) | Compute | Compute | Wt | Block |
| c_6 | cb_centered_sq | centered^2 (transient) | Compute | Compute | Wt | Block |
| c_7 | cb_var | Row variance [1,1] | Compute | Compute | 1 | Block |
| c_16 | cb_out_pre_untilize | Normalized tiles (untilize input) | Compute | Compute | Wt | Block |
| c_17 | cb_rm_out | Untilized RM sticks (writer output) | Compute | Writer | Wt | Block |
| c_24 | cb_inv_std | rsqrt(var+eps) [1,1] | Compute | Compute | 1 | Block |
| c_25 | cb_gamma | Gamma tiles [1, Wt] | Reader | Compute | Wt | Program |
| c_26 | cb_beta | Beta tiles [1, Wt] | Reader | Compute | Wt | Program |

All page sizes = tile_size (2048 bytes for bfloat16).

### Kernel Arguments

**Compile-time** (Reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * 2 bytes (row width in bytes) |
| 1+ | TensorAccessorArgs (input) | ... | Bank mapping for input buffer |

**Runtime** (Reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | num_blocks | uint32_t | Blocks this core processes |
| 2 | start_stick_id | uint32_t | First stick index for this core |
| 3 | Wt | uint32_t | Tiles per row (W/32) |
| 4 | has_gamma | uint32_t | 1 if gamma present |
| 5 | has_beta | uint32_t | 1 if beta present |
| 6 | gamma_addr | uint32_t | Gamma buffer address |
| 7 | beta_addr | uint32_t | Beta buffer address |
| 8 | eps_packed | uint32_t | Epsilon as packed bfloat16 |

**Compile-time** (Compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row |
| 1 | has_gamma | uint32_t | Gamma present flag |
| 2 | has_beta | uint32_t | Beta present flag |

**Runtime** (Compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks | uint32_t | Blocks this core processes |

**Compile-time** (Writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * 2 bytes |
| 1 | Wt | uint32_t | Tiles per row |
| 2+ | TensorAccessorArgs (output) | ... | Bank mapping for output buffer |

**Runtime** (Writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_blocks | uint32_t | Blocks this core processes |
| 2 | start_stick_id | uint32_t | First output stick index |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB (c_2) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync)
- [x] RM CBs count pages in tiles (tilize identity: Wt tiles = 32 sticks)
- [x] cb_centered (c_5) persists across variance and normalize phases -- uses WaitUpfrontNoPop/NoWaitNoPop policies

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
| c_0 (cb_rm_in) | Wt | RM-as-tile | All | Block |
| c_1 (cb_tilized) | Wt | Tile | All | Block |
| c_2 (cb_reduce_scaler) | 1 | Tile | Row0 (scaler) | Program |
| c_3 (cb_eps) | 1 | Tile | Scalar [0,0] | Program |
| c_4 (cb_mean) | 1 | Tile | Col0 (REDUCE_ROW output) | Block |
| c_5 (cb_centered) | Wt | Tile | All | Block (persistent across phases) |
| c_6 (cb_centered_sq) | Wt | Tile | All | Block (transient) |
| c_7 (cb_var) | 1 | Tile | Col0 (REDUCE_ROW output) | Block |
| c_16 (cb_out_pre_untilize) | Wt | Tile | All | Block |
| c_17 (cb_rm_out) | Wt | RM-as-tile | All | Block |
| c_24 (cb_inv_std) | 1 | Tile | Scalar | Block |
| c_25 (cb_gamma) | Wt | Tile | Row0 (broadcast ROW) | Program |
| c_26 (cb_beta) | Wt | Tile | Row0 (broadcast ROW) | Program |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| Subtract mean | SUB | All (cb_tilized) | Col0 (cb_mean) | COL |
| Square | SQUARE | All (cb_centered) | All (same CB) | NONE |
| Add eps | ADD | Col0 (cb_var) | Scalar (cb_eps) | SCALAR |
| Mul inv_std | MUL | All (cb_centered) | Scalar (cb_inv_std) | SCALAR |
| Mul gamma | MUL | All (cb_out_pre_untilize) | Row0 (cb_gamma) | ROW |
| Add beta | ADD | All (cb_out_pre_untilize) | Row0 (cb_beta) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | mean_subtract | tilize + reduce_row(mean) + sub(broadcast COL) + untilize | `x - x.mean(dim=-1, keepdim=True)` |
| 2 | full_normalize | + square + reduce_row(var) + add_eps + rsqrt + mul | Full layer_norm (no gamma/beta) |
| 3 | affine_transform | + gamma mul (ROW broadcast) + beta add (ROW broadcast) | Full layer_norm with affine |

### Stage 1: mean_subtract
- **Scope**: reader, compute, writer kernels. Phases: tilize, reduce_row(mean), sub(COL), untilize
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **CB bypass**: Compute does tilize->reduce->sub->untilize. No variance/affine phases.

### Stage 2: full_normalize
- **Scope**: compute kernel adds square, reduce_row(var), add_eps+rsqrt, mul_inv_std
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=None, bias=None, eps=1e-5)`
- **Delta from previous**: Adds phases 4-7 (square, reduce_var, add_eps+rsqrt, mul_inv_std). Reader already prepares eps scaler.
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Stage 3: affine_transform
- **Scope**: reader adds gamma/beta tile reads; compute adds mul_gamma, add_beta
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma_1d, bias=beta_1d, eps=1e-5)`
- **Delta from previous**: Adds gamma/beta reads in reader, mul(ROW) and add(ROW) phases in compute
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel
Reads RM sticks 32 at a time into cb_rm_in (same as tilize reference). At startup, prepares reduce scaler in cb_reduce_scaler via `dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler>(1.0f / W)` and epsilon tile in cb_eps via `fill_with_val`. For stage 3, reads gamma/beta tile tensors from DRAM into cb_gamma/cb_beta (Wt tiles each, once at startup).

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_rm_in, cb_reduce_scaler, cb_out_pre_untilize)` then `binary_op_init_common(cb_tilized, cb_mean, cb_out_pre_untilize)`

Per-block loop (one tile-row per iteration):

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<cb_rm_in, cb_tilized,
    tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);
```
- In: cb_rm_in [Wt tiles, waited internally by helper]
- Out: cb_tilized [Wt tiles, pushed by helper]

#### Phase 2: Reduce row (mean)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    cb_tilized, cb_reduce_scaler, cb_mean,
    ReduceInputBlockShape::row(Wt));
```
- In: cb_tilized [Wt tiles, waited upfront, NOT popped -- persists for Phase 3]
- Scaler: cb_reduce_scaler [1 tile, program lifetime, holds 1/W]
- Out: cb_mean [1 tile, pushed]

Scaler value is `1.0f/W`, so `SUM * (1/W) = mean`.

#### Phase 3: Subtract mean (broadcast COL)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_tilized, cb_mean, cb_centered,
    BinaryInputBlockShape::of(1, Wt));
```
- A: cb_tilized [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop]
- B: cb_mean [1 tile, WaitAndPopPerTile -- waited and popped per tile (1 tile total)]
- Out: cb_centered [Wt tiles, Bulk push]

**Manual pop after Phase 3**: `cb_pop_front(cb_tilized, Wt)` -- tiles no longer needed.

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| cb_tilized (c_1) | 0 | freed (manual pop) |
| cb_mean (c_4) | 0 | freed (popped by sub helper) |
| cb_centered (c_5) | Wt | freshly pushed -- persists for Phase 4 and Phase 7 |

#### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_centered, cb_centered_sq,
    BinaryInputBlockShape::of(1, Wt));
```
- In: cb_centered [Wt tiles, WaitUpfrontNoPop -- tiles persist for Phase 7]
- Out: cb_centered_sq [Wt tiles, Bulk push]

#### Phase 5: Reduce row (variance)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    cb_centered_sq, cb_reduce_scaler, cb_var,
    ReduceInputBlockShape::row(Wt));
```
- In: cb_centered_sq [Wt tiles, WaitAndPopPerTile -- consumed and freed]
- Scaler: cb_reduce_scaler [1 tile, program lifetime, holds 1/W]
- Out: cb_var [1 tile, pushed]

#### Phase 6: Add epsilon + rsqrt (fused via post_op)
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_var, cb_eps, cb_inv_std,
    BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: cb_var [1 tile, consumed and freed]
- B: cb_eps [1 tile, program lifetime, NoWaitNoPop -- already waited at kernel start]
- Out: cb_inv_std [1 tile, pushed, contains rsqrt(var+eps)]

cb_eps must have explicit `cb_wait_front(cb_eps, 1)` at kernel startup before main loop, and `cb_pop_front(cb_eps, 1)` after main loop.

#### Phase 7: Multiply by inv_std (broadcast SCALAR)
```cpp
compute_kernel_lib::mul<BroadcastDim::SCALAR,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::Bulk,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    cb_centered, cb_inv_std, cb_out_pre_untilize,
    BinaryInputBlockShape::of(1, Wt));
```
- A: cb_centered [Wt tiles, ALREADY WAITED from Phase 4, NoWaitNoPop]
- B: cb_inv_std [1 tile, consumed and freed]
- Out: cb_out_pre_untilize [Wt tiles, Bulk push]

**Manual pop after Phase 7**: `cb_pop_front(cb_centered, Wt)` -- centered tiles fully consumed.

#### Phase 8 (optional): Multiply gamma (broadcast ROW)
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop,
        BinaryOutputPolicy::Bulk,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        cb_out_pre_untilize, cb_gamma, cb_out_pre_untilize,
        BinaryInputBlockShape::of(1, Wt));
}
```
- A: cb_out_pre_untilize [Wt tiles, consumed per-tile]
- B: cb_gamma [Wt tiles, program lifetime, NoWaitNoPop -- already waited at kernel start]
- Out: cb_out_pre_untilize [Wt tiles, in-place rewrite via separate reserve/push cycle]

Note: In-place CB reuse (same input and output CB) requires that the helper consumes A tiles before pushing output. WaitAndPopPerTile for A ensures this.

#### Phase 9 (optional): Add beta (broadcast ROW)
```cpp
if constexpr (has_beta) {
    compute_kernel_lib::add<BroadcastDim::ROW,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop,
        BinaryOutputPolicy::Bulk,
        BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
        cb_out_pre_untilize, cb_beta, cb_out_pre_untilize,
        BinaryInputBlockShape::of(1, Wt));
}
```
Same pattern as Phase 8 with beta.

#### Phase 10: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_out_pre_untilize, cb_rm_out,
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::WaitBlock,
    untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
```
- In: cb_out_pre_untilize [Wt tiles, waited by helper]
- Out: cb_rm_out [Wt tiles pushed, contains 32 RM sticks]

UnpackAndPackReconfigure needed because preceding binary ops used different data format registers.

### Writer Kernel
Waits for cb_rm_out (Wt tiles = 32 RM sticks). Extracts each stick at offset `j * Wt * 32 * 2` from CB read pointer. Writes each stick to DRAM via TensorAccessor with `page_id = start_stick_id + block_idx * 32 + j`. Pops cb_rm_out after writing all 32 sticks.

### Critical Notes
1. **cb_centered persistence**: Must NOT be popped until after Phase 7. Phase 3 uses Bulk output to push all Wt tiles. Phase 4 uses WaitUpfrontNoPop. Phase 7 uses NoWaitNoPop. Manual pop after Phase 7.
2. **cb_eps and cb_reduce_scaler**: Program-lifetime CBs. Wait once at kernel start, pop once at kernel end.
3. **cb_gamma/cb_beta**: Program-lifetime. Reader loads Wt tiles once. Compute waits once at start if present, pops at end.
4. **In-place CB for gamma/beta**: cb_out_pre_untilize is both input and output for Phases 8-9. WaitAndPopPerTile on A ensures tiles are consumed before output tiles occupy the same slots.
5. **Reduce scaler value**: `1.0f / W` (not `1.0f / Wt`). This is the mean/variance divisor. PoolType::SUM with scaler `1/W` computes the mean.

### Implementation Checklist
- [ ] Reader: RM stick reads (tilize pattern), prepare_reduce_scaler, fill eps, read gamma/beta tiles
- [ ] Compute: 10 phases using helpers: tilize, reduce(SUM,REDUCE_ROW) x2, sub(COL), square, add(SCALAR)+rsqrt, mul(SCALAR), mul(ROW), add(ROW), untilize
- [ ] Writer: RM stick extraction from untilized CB (untilize writer pattern)
- [ ] CB push/pop balance verified across all phases
