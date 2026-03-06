# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), softmax W-small (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[b,h] = (1/W) * sum(input[b,h,w] for w in 0..W-1)
centered[b,h,w] = input[b,h,w] - mean[b,h]
var[b,h] = (1/W) * sum(centered[b,h,w]^2 for w in 0..W-1)
inv_std[b,h] = rsqrt(var[b,h] + epsilon)
output[b,h,w] = gamma[w] * centered[b,h,w] * inv_std[b,h] + beta[w]
```
Layer normalization along the last dimension (W) of row-major interleaved tensors, with affine parameters gamma and beta.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant added to variance |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Layout | ROW_MAJOR |
| Memory | Interleaved (DRAM) |
| Dtype | bfloat16 |
| Shape | At least 2D; last 2 dims must be multiples of 32 (tile-aligned) |

Gamma and beta: shape `(1, 1, 1, W)`, same dtype and layout as input.

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: bfloat16 (same as input)
- **Layout**: ROW_MAJOR
- **Memory**: Interleaved (DRAM)

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt = 1, all phases still apply |
| All-zero input row | mean=0, var=0, output = beta (rsqrt(eps) * gamma * 0 + beta) |
| Large W (many tiles per row) | Full row must fit in L1; CB sized to Wt tiles |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add gamma/beta/scaler CB generation; read 3 RM tensors |
| Compute | softmax (multi-phase pattern) + NEW | compute_core | Replace 4-phase softmax with 7-phase layer norm; use helpers |
| Writer | untilize | output_stage | None (standard RM stick writer) |

### Work Distribution
- **Work unit**: tile-row (one row of Wt tiles, spanning 32 element-rows of width W)
- **Grid**: 1D linearized, up to `device.compute_with_storage_grid_size()`
- **Work per core**: `N = ceil(total_tile_rows / num_cores)` tile-rows
- **Remainder**: Two-group split (core_group_1 gets ceil, core_group_2 gets floor)

Total tile-rows = `(product of all dims except last two) * H / 32`.

### Data Flow
Reader reads 32 RM sticks into cb_in (input), reads gamma/beta row tiles once. Compute tilizes input to cb_tilized, runs 7 compute phases (reduce_mean, subtract_mean, square, reduce_var, rsqrt, normalize, affine), untilizes result to cb_untilized. Writer writes RM sticks to DRAM output.

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in | RM input sticks | Reader | Compute (tilize) | Wt | Block |
| c_1 | cb_gamma | Gamma row (tiled) | Reader (once) | Compute (Phase 7) | Wt | Program |
| c_2 | cb_beta | Beta row (tiled) | Reader (once) | Compute (Phase 7) | Wt | Program |
| c_8 | cb_reduce_scaler | Reduce scaler (1/W) | Reader (once) | Compute (reduce phases) | 1 | Program |
| c_9 | cb_eps | Epsilon tile | Reader (once) | Compute (Phase 4) | 1 | Program |
| c_16 | cb_out | RM output sticks (untilized) | Compute (untilize) | Writer | Wt | Block |
| c_24 | cb_tilized | Tilized input tiles | Compute (tilize) | Compute (Phase 1,2) | Wt | Row |
| c_25 | cb_mean | Mean tile (reduce result) | Compute (Phase 1) | Compute (Phase 2) | 1 | Row |
| c_26 | cb_centered | x - mean | Compute (Phase 2) | Compute (Phase 3,6) | Wt | Row |
| c_27 | cb_squared | (x - mean)^2 | Compute (Phase 3) | Compute (Phase 4) | Wt | Row |
| c_28 | cb_var_eps | var + eps | Compute (Phase 4) | Compute (Phase 5) | 1 | Row |
| c_29 | cb_inv_std | rsqrt(var + eps) | Compute (Phase 5) | Compute (Phase 6) | 1 | Row |
| c_30 | cb_normed | centered * inv_std | Compute (Phase 6) | Compute (Phase 7) | Wt | Row |
| c_31 | cb_affine_out | gamma * normed + beta = final tiled output | Compute (Phase 7) | Compute (untilize) | Wt | Row |

### Kernel Arguments

**Compile-time** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | Width of one RM stick in bytes |
| Reader | 1+ | input_accessor_args | uint32_t[] | TensorAccessor args for input |
| Compute | 0 | N | uint32_t | Number of tile-rows this core processes |
| Compute | 1 | Wt | uint32_t | Tiles per tile-row (width) |
| Writer | 0 | cb_id_out | uint32_t | Output CB index (c_16) |
| Writer | 1 | output_stick_size | uint32_t | Bytes per output stick |
| Writer | 2 | tile_height | uint32_t | 32 |
| Writer | 3 | num_tiles_per_row | uint32_t | Wt |
| Writer | 4+ | output_accessor_args | uint32_t[] | TensorAccessor args for output |

**Runtime** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer base address |
| Reader | 1 | N | uint32_t | Tile-rows for this core |
| Reader | 2 | start_stick_id | uint32_t | First stick ID for this core |
| Reader | 3 | gamma_addr | uint32_t | Gamma buffer base address |
| Reader | 4 | beta_addr | uint32_t | Beta buffer base address |
| Writer | 0 | dst_addr | uint32_t | Output buffer base address |
| Writer | 1 | N | uint32_t | Tile-rows for this core |
| Writer | 2 | start_stick_id | uint32_t | First output stick ID |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB is bfloat16 (cb_reduce_scaler uses input data format)
- [x] DEST register holds max 8 tiles (bf16 half-sync) / 4 tiles (f32 half-sync)
- [x] RM CBs count pages in tiles (tile_size page format for tilize compatibility)
- [x] cb_in page_size = tile_size, num_pages = Wt (32 sticks = Wt tiles of data)

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
| c_0 (cb_in) | Wt | tile-sized (holds RM data) | All | Block |
| c_1 (cb_gamma) | Wt | tile | Row (ROW bcast) | Program |
| c_2 (cb_beta) | Wt | tile | Row (ROW bcast) | Program |
| c_8 (cb_reduce_scaler) | 1 | tile | Col0 (scaler) | Program |
| c_9 (cb_eps) | 1 | tile | [0,0] (scalar) | Program |
| c_16 (cb_out) | Wt | tile-sized (holds RM data) | All | Block |
| c_24 (cb_tilized) | Wt | tile | All | Row |
| c_25 (cb_mean) | 1 | tile | Col0 | Row |
| c_26 (cb_centered) | Wt | tile | All | Row |
| c_27 (cb_squared) | Wt | tile | All | Row |
| c_28 (cb_var_eps) | 1 | tile | Col0 | Row |
| c_29 (cb_inv_std) | 1 | tile | Col0 | Row |
| c_30 (cb_normed) | Wt | tile | All | Row |
| c_31 (cb_affine_out) | Wt | tile | All | Row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 2: sub mean | SUB | All (cb_tilized) | Col0 (cb_mean: REDUCE_ROW output) | COL |
| 4: add eps | ADD | Col0 (cb_var_eps: REDUCE_ROW output) | [0,0] (cb_eps: scalar) | SCALAR |
| 6: mul inv_std | MUL | All (cb_centered) | Col0 (cb_inv_std: scalar-like) | COL |
| 7a: mul gamma | MUL | All (cb_normed) | Row (cb_gamma: [1,W]) | ROW |
| 7b: add beta | ADD | All (after gamma*normed) | Row (cb_beta: [1,W]) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader + tilize + untilize + writer (identity passthrough) | `input` (passthrough) |
| 2 | reduce_mean | Phase 1: reduce SUM with 1/W scaler | `x.mean(dim=-1, keepdim=True).expand_as(x)` broadcast to full shape |
| 3 | subtract_mean | Phase 2: sub mean from input with COL broadcast | `x - x.mean(dim=-1, keepdim=True)` |
| 4 | variance | Phases 3-4: square + reduce SUM for variance | `((x - x.mean(-1,True))**2).mean(-1,True).expand_as(x)` |
| 5 | normalize | Phase 5-6: rsqrt(var+eps), multiply centered | Full normalization without affine |
| 6 | affine | Phase 7: gamma * normed + beta | Full layer_norm with affine |

### Stage 1: data_pipeline
- **Scope**: reader kernel (RM stick reader for input only), compute (tilize + untilize only), writer kernel (RM stick writer)
- **Reference**: `return input_tensor` (identity passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes cb_in -> cb_tilized, then immediately untilizes cb_tilized -> cb_out. All intermediate phases skipped.

### Stage 2: reduce_mean
- **Scope**: Add reduce scaler generation in reader. Add Phase 1 (reduce SUM REDUCE_ROW with 1/W scaler). Output is mean broadcast to full width via COL broadcast mul by ones, then untilize.
- **Reference**: `return input_tensor.to(torch.float32).mean(dim=-1, keepdim=True).expand_as(input_tensor).to(torch.bfloat16)` -- mean replicated to full shape
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Adds reduce helper call, scaler CB setup

### Stage 3: subtract_mean
- **Scope**: Add Phase 2 (sub with COL broadcast). Output is centered values.
- **Reference**: `return (input_tensor.to(torch.float32) - input_tensor.to(torch.float32).mean(dim=-1, keepdim=True)).to(torch.bfloat16)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Adds binary sub helper with COL broadcast

### Stage 4: variance
- **Scope**: Add Phases 3-4 (square centered + reduce SUM for variance). Output is variance broadcast to full shape.
- **Reference**: `x = input_tensor.to(torch.float32); c = x - x.mean(-1, True); return (c**2).mean(-1, True).expand_as(x).to(torch.bfloat16)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Adds square helper + second reduce

### Stage 5: normalize
- **Scope**: Add Phases 5-6 (add eps, rsqrt, multiply centered by inv_std). Output is fully normalized (no affine).
- **Reference**: `return torch.nn.functional.layer_norm(input_tensor.to(torch.float32), [input_tensor.shape[-1]], eps=1e-5).to(torch.bfloat16)` -- layer_norm without affine (weight=None, bias=None)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Adds eps CB, add helper, rsqrt post-op, mul COL broadcast

### Stage 6: affine
- **Scope**: Add Phase 7 (mul gamma ROW broadcast + add beta ROW broadcast). Full layer_norm output.
- **Reference**: Full `layer_norm` with gamma and beta
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Adds gamma/beta reader logic, two binary ops with ROW broadcast

### Reader Kernel
Reads RM sticks from interleaved DRAM using TensorAccessor (tilize pattern: batch 32 sticks per block, push Wt pages). On first iteration only, generates: (1) reduce scaler tile in cb_reduce_scaler via `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, SUM, REDUCE_ROW, 32, W>()` where W is the reduce factor for AVG-equivalent behavior, (2) epsilon scalar tile in cb_eps via `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(epsilon)`, (3) reads gamma/beta tiles from DRAM into cb_gamma/cb_beta (Wt tiles each, tile-layout tensors, read once and never popped).

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in, cb_reduce_scaler, cb_out);`

Wait for constant CBs before main loop:
```cpp
cb_wait_front(cb_reduce_scaler, 1);
cb_wait_front(cb_eps, 1);
cb_wait_front(cb_gamma, Wt);
cb_wait_front(cb_beta, Wt);
```

Per tile-row iteration (N times):

#### Phase 1: Tilize input
```cpp
compute_kernel_lib::tilize<cb_in, cb_tilized>(Wt, 1);
```
- In: cb_in [Wt pages of RM data, pushed by reader]
- Out: cb_tilized [Wt tiles, pushed by tilize helper]

#### Phase 2: Reduce mean (SUM with 1/W scaler)
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_tilized, cb_reduce_scaler, cb_mean,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- A: cb_tilized [Wt tiles, FRESHLY PUSHED by Phase 1, NoPop -- persists for Phase 3]
- B: cb_reduce_scaler [1 tile, program-lifetime constant]
- Out: cb_mean [1 tile, pushed by reduce helper]

#### Phase 3: Subtract mean (COL broadcast)
```cpp
compute_kernel_lib::sub<compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    cb_tilized, cb_mean, cb_centered,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_tilized [Wt tiles, ALREADY WAITED from Phase 2 via WaitUpfrontNoPop, NoWaitNoPop -- caller pops after]
- B: cb_mean [1 tile, freshly pushed by Phase 2, popped by helper (COL bcast pops per row)]
- Out: cb_centered [Wt tiles, pushed]

Manual pop after Phase 3:
```cpp
cb_pop_front(cb_tilized, Wt);
```

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| cb_tilized | 0 | freed (manually popped) |
| cb_mean | 0 | freed (popped by sub helper) |
| cb_centered | Wt | freshly pushed -- persists for Phase 6 |

#### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_centered, cb_squared,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_centered [Wt tiles, FRESHLY PUSHED by Phase 3, NoPop -- persists for Phase 6]
- Out: cb_squared [Wt tiles, pushed]

#### Phase 5: Reduce variance (SUM with 1/W scaler)
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
    cb_squared, cb_reduce_scaler, cb_var_eps,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt),
    compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
    compute_kernel_lib::NoAccumulation{},
    [](uint32_t dst_idx) {});
```
- A: cb_squared [Wt tiles, freshly pushed by Phase 4, popped by reduce helper (WaitAndPopPerTile default)]
- Out: cb_var_eps [1 tile, pushed]

Note: cb_var_eps is reused for the add-epsilon result. The reduce outputs the raw variance; epsilon is added next.

#### Phase 6: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::SCALAR,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
    cb_var_eps, cb_eps, cb_inv_std,
    compute_kernel_lib::BinaryInputBlockShape::single(),
    {},
    compute_kernel_lib::NoAccumulation{},
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: cb_var_eps [1 tile, freshly pushed by Phase 5, popped by helper]
- B: cb_eps [1 tile, program-lifetime constant, NoWaitNoPop]
- Out: cb_inv_std [1 tile, after add+rsqrt post-op]

#### Phase 7: Multiply centered by inv_std (COL broadcast)
```cpp
compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    cb_centered, cb_inv_std, cb_normed,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_centered [Wt tiles, ALREADY WAITED from Phase 4, NoWaitNoPop -- caller pops after]
- B: cb_inv_std [1 tile, freshly pushed by Phase 6, popped by helper (COL bcast)]
- Out: cb_normed [Wt tiles, pushed]

Manual pop after Phase 7:
```cpp
cb_pop_front(cb_centered, Wt);
```

#### Phase 8: Affine transform (gamma * normed + beta)

Gamma multiply (ROW broadcast):
```cpp
compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::ROW,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
    cb_normed, cb_gamma, cb_affine_out,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: cb_normed [Wt tiles, freshly pushed by Phase 7, popped by helper]
- B: cb_gamma [Wt tiles, program-lifetime, NoWaitNoPop]
- Out: cb_affine_out [Wt tiles, pushed]

Beta add (ROW broadcast):
```cpp
compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::ROW,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
    cb_affine_out, cb_beta, cb_affine_out_final,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```

Note: cb_affine_out cannot be both input and output. Use a ping-pong: output gamma*normed to cb_affine_out, then add beta reading from cb_affine_out writing to a separate CB. Since we are about to untilize, write beta result directly to cb_31 if using cb_affine_out=c_30 for gamma result. Revised: gamma*normed outputs to cb_normed (reuse since normed is consumed), beta add reads cb_normed and outputs to cb_affine_out.

Corrected Phase 8:
```cpp
// gamma * normed -> cb_normed (reused, was already popped by mul)
// Actually cb_normed is the input to gamma mul, it gets consumed.
// Output of gamma mul goes to cb_affine_out (c_31).
// Then beta add: cb_affine_out (c_31) -> need separate output.
```

Final design for Phase 8: Use cb_30 for gamma*normed output, cb_31 for final output after +beta.

```cpp
// 8a: gamma * normed -> cb_30 (cb_normed reused as output, since normed tiles are consumed by mul)
compute_kernel_lib::mul<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    cb_normed, cb_gamma, cb_affine_out, BinaryInputBlockShape::row(Wt));

// 8b: (gamma * normed) + beta -> cb_31
compute_kernel_lib::add<BroadcastDim::ROW,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    cb_affine_out, cb_beta, cb_final, BinaryInputBlockShape::row(Wt));
```

Wait -- cb_30 is cb_normed and cb_31 is cb_affine_out in our CB layout. The gamma mul consumes cb_normed (c_30) and writes to cb_affine_out (c_31). Then beta add consumes cb_affine_out (c_31). We need another CB for the final output before untilize. To avoid needing yet another CB, merge gamma*normed+beta into a single phase using post_op on the mul, or use cb_normed (c_30) again since it was freed by the mul.

Revised: gamma mul: cb_normed(c_30) -> cb_affine_out(c_31). Beta add: cb_affine_out(c_31) -> cb_normed(c_30) [reuse freed CB]. Then untilize from cb_normed(c_30).

```cpp
// 8a: gamma * normed -> c_31
mul<ROW, WaitAndPopPerTile, NoWaitNoPop>(cb_normed, cb_gamma, cb_affine_out, row(Wt));
// 8b: + beta -> c_30 (reuse freed cb_normed)
add<ROW, WaitAndPopPerTile, NoWaitNoPop>(cb_affine_out, cb_beta, cb_normed, row(Wt));
```

Then untilize from cb_normed(c_30) -> cb_out(c_16).

#### Phase 9: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_normed, cb_out>(1);
```
- In: cb_normed [Wt tiles, pushed by Phase 8b]
- Out: cb_out [Wt pages of RM data, for writer]

### Writer Kernel
Standard RM stick writer from untilize reference. Per block: waits for Wt pages in cb_out, extracts 32 rows, writes each row to DRAM output via TensorAccessor with `noc_async_write`. Barrier per block. Uses the untilize writer pattern: `get_read_ptr(cb_out)` to get L1 base address, then offset by `row * stick_size` for each of the 32 sticks.

### Critical Notes
1. **cb_tilized persistence**: Phase 2 reduce uses WaitUpfrontNoPop so tiles persist for Phase 3 subtract. Manual `cb_pop_front(cb_tilized, Wt)` required after Phase 3.
2. **cb_centered persistence**: Phase 4 square uses WaitUpfrontNoPop so centered tiles persist for Phase 7 normalize. Manual `cb_pop_front(cb_centered, Wt)` required after Phase 7.
3. **Constant CBs**: cb_reduce_scaler, cb_eps, cb_gamma, cb_beta are written once by reader and never popped. Waited at top of compute loop.
4. **cb_eps policy**: Phase 6 add uses NoWaitNoPop for B (cb_eps), since it was already waited at program start and is never popped.
5. **cb_gamma/cb_beta policy**: Phase 8a/8b mul/add use NoWaitNoPop for B, since they were waited at program start and are never popped.
6. **Reduce scaler value**: For mean computation (SUM with 1/W scaler), use `calculate_and_prepare_reduce_scaler<cb_reduce_scaler, SUM, REDUCE_ROW, 32, W>()` where W is the tensor width in elements. This produces scaler = 1/W. The `tile_columns_to_fill=32` since we assume tile-aligned width.
7. **Ping-pong for affine**: Phase 8a outputs to c_31, Phase 8b reads c_31 and outputs to c_30 (freed after gamma mul consumed it). Untilize reads from c_30.
8. **Init/Uninit chaining**: tilize and untilize use InitAndUninit (default) since they are separated by binary/reduce ops that change hardware config.

### Implementation Checklist
- [ ] Reader: TensorAccessor for input sticks (32-stick batching), gamma/beta tile reads, scaler/eps generation
- [ ] Compute: 9 phases using helpers: tilize, reduce (x2), sub, square, add (x2), mul (x2), untilize
- [ ] Writer: TensorAccessor for output sticks (row-by-row DRAM writes)
- [ ] CB push/pop balance verified (manual pops for cb_tilized and cb_centered)
