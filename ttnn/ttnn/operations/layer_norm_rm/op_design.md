# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), untilize (output_stage), softmax_general_w (compute_core)

## Mathematical Definition
```
mean[b,h] = sum(x[b,h,:]) / W
centered[b,h,w] = x[b,h,w] - mean[b,h]
var[b,h] = sum(centered[b,h,:]^2) / W
inv_std[b,h] = rsqrt(var[b,h] + epsilon)
x_norm[b,h,w] = centered[b,h,w] * inv_std[b,h]
output[b,h,w] = gamma[w] * x_norm[b,h,w] + beta[w]
```
Layer normalization over the last dimension (W). Each row is independently normalized, then affine-transformed by gamma (scale) and beta (bias).

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-6 | Numerical stability for rsqrt |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | bfloat16 | "Input must be bfloat16" |
| layout | ROW_MAJOR | "Input must be row-major" |
| memory | Interleaved DRAM | "Input must be interleaved" |
| rank | >= 2 | "Need at least 2 dimensions" |
| last 2 dims | Multiples of 32 | "H and W must be tile-aligned" |

Gamma: bfloat16, RM, shape (1,1,1,W). Beta: bfloat16, RM, shape (1,1,1,W).

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: bfloat16
- **Layout**: ROW_MAJOR
- **Memory**: Interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce is single-tile |
| Large W (many tiles) | L1 must fit ~7*Wt + 4 tiles of intermediates |
| epsilon = 0 | Allowed but risks division by zero |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize + softmax | input_stage | Read RM sticks (tilize pattern) + scaler/gamma/beta (softmax pattern) |
| Compute | softmax w_small | compute_core | Replace max/exp/recip with mean/square/rsqrt; add tilize/untilize phases |
| Writer | untilize | output_stage | Extract RM sticks from untilized output CB |

### Work Distribution
- **Work unit**: 1 tile-row = 32 rows of the tensor = Wt tiles across width
- **Grid**: 1D linearized, up to `grid_size.x * grid_size.y` cores
- **Work per core**: `ceil(nblocks / ncores)` tile-rows, where `nblocks = H_total / 32`
- **Remainder**: Cliff core gets `nblocks % nblocks_per_core` tile-rows

### Data Flow
Reader reads 32 RM sticks per tile-row and pushes to input CB. Compute tilizes, performs full layer norm pipeline (mean, subtract, square, variance, rsqrt, normalize, affine), then untilizes. Writer extracts 32 RM sticks from output CB and writes to DRAM. Reader also provides scaler, epsilon, gamma, and beta tiles (once at startup).

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in_rm | RM sticks for tilize | Reader | Compute | Wt | Per block |
| c_1 | cb_tilized | Tilized input | Compute | Compute | Wt | Phases 1-3 |
| c_2 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | Program |
| c_3 | cb_eps | Epsilon scalar tile | Reader | Compute | 1 | Program |
| c_4 | cb_gamma | Gamma affine tiles | Reader | Compute | Wt | Program |
| c_5 | cb_beta | Beta affine tiles | Reader | Compute | Wt | Program |
| c_16 | cb_out_rm | Untilized output | Compute | Writer | Wt | Per block |
| c_24 | cb_mean | Row mean | Compute | Compute | 1 | Phases 2-3 |
| c_25 | cb_centered | x - mean | Compute | Compute | Wt | Phases 3-6 |
| c_26 | cb_centered_sq | (x - mean)^2 | Compute | Compute | Wt | Phases 4-5 |
| c_27 | cb_inv_std | rsqrt(var + eps) | Compute | Compute | 1 | Phases 5-6 |
| c_28 | cb_normed | x_norm | Compute | Compute | Wt | Phases 6-7 |
| c_29 | cb_scaled | gamma * x_norm | Compute | Compute | Wt | Phases 7-8 |

All tile-format CBs use `tile_size` as page size. c_0 uses `tile_size` pages but is filled with RM sticks (same byte count: 32 sticks * W_bytes = Wt * tile_size). c_16 holds untilized data in tile-sized pages.

### Kernel Arguments

**Compile-time** (per kernel):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | stick_size | uint32_t | W * element_size bytes |
| Reader | 1+ | TensorAccessorArgs(input) | uint32_t[] | Input tensor accessor |
| Compute | 0 | Wt | uint32_t | Tiles per row (W / 32) |
| Compute | 1 | num_rows | uint32_t | Tile-rows this core processes |
| Writer | 0 | stick_size | uint32_t | W * element_size bytes |
| Writer | 1 | Wt | uint32_t | Tiles per row |
| Writer | 2+ | TensorAccessorArgs(output) | uint32_t[] | Output tensor accessor |

**Runtime** (per kernel):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | src_addr | uint32_t | Input buffer base address |
| Reader | 1 | num_blocks | uint32_t | Tile-rows for this core |
| Reader | 2 | start_stick_id | uint32_t | First stick index |
| Reader | 3 | gamma_addr | uint32_t | Gamma buffer base address |
| Reader | 4 | beta_addr | uint32_t | Beta buffer base address |
| Reader | 5 | eps_value | uint32_t | Epsilon as bit-cast uint32 |
| Reader | 6 | mean_scaler_value | uint32_t | 1/W as bit-cast uint32 |
| Writer | 0 | dst_addr | uint32_t | Output buffer base address |
| Writer | 1 | num_blocks | uint32_t | Tile-rows for this core |
| Writer | 2 | start_stick_id | uint32_t | First output stick index |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count per phase
- [x] Reduce scaler CB (c_2) is bfloat16 format
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- reduce and binary_op helpers handle chunking
- [x] RM CBs count pages in tiles (byte-equivalent), tile CBs count in tiles

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` with specified rtol/atol

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration along W | `(1, 1, 32, 128)` |
| Multi-tile HW | Multiple tile-rows | `(1, 1, 64, 128)` |
| Non-square | W != H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (cb_in_rm) | Wt | RM-as-tile | All (RM sticks) | Per block |
| c_1 (cb_tilized) | Wt | Tile | All | Phases 1-3 |
| c_2 (cb_scaler) | 1 | Tile | Row0 faces | Program |
| c_3 (cb_eps) | 1 | Tile | Scalar [0,0] | Program |
| c_4 (cb_gamma) | Wt | Tile | All | Program |
| c_5 (cb_beta) | Wt | Tile | All | Program |
| c_16 (cb_out_rm) | Wt | RM-as-tile | All | Per block |
| c_24 (cb_mean) | 1 | Tile | Col0 | Phases 2-3 |
| c_25 (cb_centered) | Wt | Tile | All | Phases 3-6 |
| c_26 (cb_centered_sq) | Wt | Tile | All | Phases 4-5 |
| c_27 (cb_inv_std) | 1 | Tile | Col0 | Phases 5-6 |
| c_28 (cb_normed) | Wt | Tile | All | Phases 6-7 |
| c_29 (cb_scaled) | Wt | Tile | All | Phases 7-8 |

After Phase 8, the result (gamma*x_norm + beta) must go to a CB for untilize input. c_1 is freed after Phase 3 and can be reused as the Phase 8 output / Phase 9 untilize input.

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 3 (sub mean) | SUB | All (cb_tilized) | Col0 (cb_mean) | COL |
| 6 (mul inv_std) | MUL | All (cb_centered) | Col0 (cb_inv_std) | COL |
| 7 (mul gamma) | MUL | All (cb_normed) | All (cb_gamma) | NONE |
| 8 (add beta) | ADD | All (cb_scaled) | All (cb_beta) | NONE |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | mean_subtract | Tilize + reduce mean + sub mean + untilize | `x - x.mean(dim=-1, keepdim=True)` |
| 2 | variance_normalize | Square + reduce var + eps + rsqrt + mul | Full standardization (no affine) |
| 3 | affine_transform | Gamma mul + beta add | Full layer_norm with gamma/beta |

### Stage 1: mean_subtract
- **Scope**: reader (input + scaler), compute (tilize, reduce, sub, untilize), writer (RM output)
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.05
- **Active CBs**: c_0, c_1, c_2, c_16, c_24, c_25 (c_25 used as untilize input since no gamma/beta yet)
- **Compute phases**: 1 (tilize), 2 (reduce mean), 3 (sub mean), 4 (untilize c_25 directly)

### Stage 2: variance_normalize
- **Scope**: compute adds square, variance reduce with rsqrt post-op, mul inv_std
- **Reference**: `(x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, correction=0) + 1e-6)`
- **Delta from Stage 1**: Adds Phases 4-6 (square, reduce+rsqrt, mul), epsilon CB (c_3). Untilize now reads from cb_normed (c_28) instead of cb_centered (c_25).
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Stage 3: affine_transform
- **Scope**: reader adds gamma/beta tile reads, compute adds mul gamma + add beta phases
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma.flatten(), bias=beta.flatten(), eps=1e-6)`
- **Delta from Stage 2**: Adds gamma/beta CBs (c_4, c_5), Phases 7-8 (mul gamma, add beta). Untilize reads from c_1 (reused as Phase 8 output).
- **Shapes**: `(1,1,32,32)`, `(1,1,32,128)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel
Reads RM sticks from DRAM using TensorAccessor. At startup, generates reduce scaler tile (1/W) into c_2 via `prepare_reduce_scaler`, and epsilon scalar tile into c_3. For Stage 3, also reads gamma/beta as tilized tile rows into c_4/c_5 (reader tilizes gamma/beta sticks by reading Wt-tiles-worth of stick data). Main loop: for each tile-row, reserves c_0 for Wt pages, reads 32 sticks via noc_async_read, pushes Wt pages. Pattern matches tilize reference.

### Compute Kernel

**Startup**: `binary_op_init_common(c_1, c_2, c_16)` then `compute_kernel_hw_startup(c_1, c_2, c_16)`

Outer loop: N tile-rows. Each iteration:

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<c_0, c_1,
    tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode::WaitBlock>(Wt, 1);
```
- Input: c_0 [Wt pages, pushed by reader]
- Output: c_1 [Wt tiles, tilized]

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| c_0 | 0 | freed (popped by tilize helper) |
| c_1 | Wt | pushed by tilize helper |

#### Phase 2: Reduce SUM for mean
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop>(
    c_1, c_2, c_24,
    ReduceInputBlockShape::row(Wt));
```
- A: c_1 [Wt tiles, FRESHLY PUSHED from Phase 1, WaitUpfrontNoPop -- tiles persist for Phase 3]
- Scaler: c_2 [1 tile, value=1/W, never popped]
- Out: c_24 [1 tile, mean as col vector]

Note: scaler=1/W makes SUM behave as AVG. The reduce helper does `sum(tiles) * scaler = sum/W = mean`.

#### Phase 3: Subtract mean (x - mean)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_1, c_24, c_25,
    BinaryInputBlockShape::row(Wt));
```
- A: c_1 [Wt tiles, ALREADY WAITED from Phase 2, WaitUpfrontPopAtEnd -- consumed here]
- B: c_24 [1 tile, mean col vector, WaitUpfrontPopAtEnd -- consumed here]
- Out: c_25 [Wt tiles, x - mean]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | 0 | freed (popped by sub helper) |
| c_24 | 0 | freed (popped by sub helper) |
| c_25 | Wt | freshly pushed |

#### Phase 4: Square (x - mean)^2
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_25, c_26,
    BinaryInputBlockShape::row(Wt));
```
- A: c_25 [Wt tiles, FRESHLY PUSHED from Phase 3, WaitUpfrontNoPop -- persists for Phase 6]
- Out: c_26 [Wt tiles, (x-mean)^2]

#### Phase 5: Reduce variance + add eps + rsqrt
```cpp
compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile>(
    c_26, c_2, c_27,
    ReduceInputBlockShape::row(Wt),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst_idx) {
        // Post-reduce: var is in DST[dst_idx]
        // Add epsilon: load eps tile, add to DST
        // This requires manual tile ops -- see Critical Notes
    });
```

**Epsilon handling**: The post-reduce lambda cannot easily add epsilon from a CB (requires unpack). Instead, split into sub-phases:
- Phase 5a: Reduce SUM with scaler=1/W -> c_27 (variance)
- Phase 5b: `add<BroadcastDim::SCALAR>(c_27, c_3, c_27_out, single())` to add epsilon. Since c_27 is both input and output, we need a temp CB. Use c_24 (freed in Phase 3) as temp:
  `add<SCALAR>(c_27, c_3, c_24, single())` -> c_24 holds var+eps

Actually, we need to be more careful. After reduce outputs to c_27, we need to add eps. The binary add helper reads from two input CBs and writes to an output CB. We can use c_24 as the output (it was freed in Phase 3).

Then rsqrt: no helper exists for unary rsqrt. We handle it via the binary `mul` post-op or manual DST ops. Cleanest approach: use `square` as a self-mul with a post-op lambda:

```cpp
// Copy var+eps to DST, apply rsqrt
compute_kernel_lib::mul<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_24, c_24_dummy, c_27, BinaryInputBlockShape::single(), {},
    NoAccumulation{},
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```

This is awkward. Better approach: manual DST ops for rsqrt on a single tile:
```cpp
cb_wait_front(c_24, 1);  // var+eps
cb_reserve_back(c_27, 1);
tile_regs_acquire();
copy_tile_init_with_dt(c_24);
copy_tile(c_24, 0, 0);   // load to DST[0]
rsqrt_tile_init();
rsqrt_tile(0);            // rsqrt in-place
tile_regs_commit();
tile_regs_wait();
pack_tile_with_dt(0, c_27);
tile_regs_release();
cb_pop_front(c_24, 1);
cb_push_back(c_27, 1);
```

**Revised Phase 5 plan**:
- 5a: `reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>(c_26, c_2, c_24, row(Wt))` -> c_24 = variance (c_26 freed)
- 5b: `add<SCALAR>(c_24, c_3, c_27, single())` -> c_27 = var + eps (c_24 freed)
- 5c: Manual rsqrt on c_27 -> overwrite c_27 with inv_std. Use c_24 as temp output:
  Manual copy_tile + rsqrt from c_27 -> c_24. Then c_27 freed, c_24 = inv_std.

For clarity, let's rename: after Phase 5c, the inv_std lives in c_24. We use c_24 as cb_inv_std for Phase 6.

**Revised CB flow**:
- Phase 5a: reduce -> c_24 (var). c_26 freed.
- Phase 5b: add eps -> c_27 (var+eps). c_24 freed.
- Phase 5c: rsqrt -> c_24 (inv_std). c_27 freed. (Manual: read c_27, rsqrt, write c_24)

#### Phase 6: Normalize (x - mean) * inv_std
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_25, c_24, c_28,
    BinaryInputBlockShape::row(Wt));
```
- A: c_25 [Wt tiles, ALREADY WAITED from Phase 4, WaitUpfrontPopAtEnd -- consumed]
- B: c_24 [1 tile, inv_std col vector, WaitUpfrontPopAtEnd -- consumed]
- Out: c_28 [Wt tiles, x_norm]

**CB state after Phase 6:**
| CB | Tiles | State |
|----|-------|-------|
| c_25 | 0 | freed |
| c_24 | 0 | freed |
| c_28 | Wt | freshly pushed |

#### Phase 7: Multiply by gamma
```cpp
compute_kernel_lib::mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_28, c_4, c_29,
    BinaryInputBlockShape::row(Wt));
```
- A: c_28 [Wt tiles, WaitAndPopPerTile -- consumed tile by tile]
- B: c_4 [Wt tiles, gamma, NoWaitNoPop -- persistent, already waited at program start]
- Out: c_29 [Wt tiles, gamma * x_norm]

#### Phase 8: Add beta
```cpp
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_29, c_5, c_1,
    BinaryInputBlockShape::row(Wt));
```
- A: c_29 [Wt tiles, WaitAndPopPerTile -- consumed]
- B: c_5 [Wt tiles, beta, NoWaitNoPop -- persistent]
- Out: c_1 [Wt tiles, final result -- c_1 reused, freed since Phase 3]

#### Phase 9: Untilize
```cpp
compute_kernel_lib::untilize<Wt, c_1, c_16,
    untilize_config::InitUninitMode::InitAndUninit,
    untilize_config::WaitMode::NoWait>(1);
```
- Input: c_1 [Wt tiles, NoWait -- data already in CB from Phase 8]
- Output: c_16 [Wt tiles, untilized RM data]

Note: `NoWait` because Phase 8 already pushed data into c_1. The untilize helper skips cb_wait_front.

### Writer Kernel
Waits on c_16 (Wt tiles per block), extracts 32 RM sticks using `get_read_ptr` + stride calculation, writes each stick to DRAM via TensorAccessor. Pattern matches untilize reference: `noc_async_write` per stick, barrier after all 32 sticks, then `cb_pop_front(c_16, Wt)`.

### Critical Notes

1. **Gamma/beta are RM but must be tilized for compute**: Reader must tilize gamma/beta sticks into tile format. Read gamma as 1 stick of W elements, then push as tilized tiles. Since gamma shape is (1,1,1,W), it's a single stick. The reader needs to read this one stick and produce Wt tiles. Use the tilize compute phase OR read gamma/beta as already-tilized (host can pre-tilize). Simplest: host tilizes gamma/beta before dispatch, reader reads tiles directly.

2. **Scaler value is 1/W**: Generated at runtime since W varies. Reader computes `1.0f / W` and calls `prepare_reduce_scaler<c_2>(1.0f / W)`. The scaler must be bfloat16 format.

3. **Epsilon scalar tile**: Reader generates via `prepare_reduce_scaler<c_3>(epsilon)`. This fills row0 of each face with epsilon, suitable for `add<SCALAR>`.

4. **Phase 5c rsqrt is manual**: No unary helper exists. Kernel writer must implement copy_tile + rsqrt_tile + pack_tile manually with DST register management.

5. **CB c_1 reuse**: c_1 is used for tilized input (Phases 1-3) and reused as Phase 8 output / Phase 9 untilize input. The kernel writer must ensure c_1 is fully freed before Phase 8 writes to it.

6. **Gamma/beta NoWaitNoPop**: These CBs are filled once at startup by the reader and never popped. Compute accesses them via indexed tile access every iteration. The reader must `cb_wait_front` on gamma/beta at startup (or rely on compute's first wait). Since the reader pushes them before the main loop and compute waits in Phase 7/8, synchronization is automatic.

### Implementation Checklist
- [ ] Reader: TensorAccessor for RM input sticks, prepare_reduce_scaler for c_2 and c_3, read gamma/beta tiles into c_4/c_5
- [ ] Compute: 9 phases using helpers: tilize, reduce(SUM,ROW), sub(COL), square, reduce(SUM,ROW), add(SCALAR), manual rsqrt, mul(COL), mul(NONE), add(NONE), untilize
- [ ] Writer: TensorAccessor for RM output sticks, 32-stick extraction per block
- [ ] CB push/pop balance verified per phase
