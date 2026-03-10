# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), softmax (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[b,h] = sum(x[b,h,:]) / W
centered[b,h,w] = x[b,h,w] - mean[b,h]
var[b,h] = sum(centered[b,h,:]^2) / W
output[b,h,w] = centered[b,h,w] * rsqrt(var[b,h] + epsilon)
if gamma: output[b,h,w] *= gamma[w]
if beta:  output[b,h,w] += beta[w]
```
Per-row normalization over the last dimension. Each row of 32 elements (tile height) is independently normalized.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Variance stabilization constant |
| gamma | Tensor | No | shape (1,1,1,W) bf16 RM | None | Scale parameter |
| beta | Tensor | No | shape (1,1,1,W) bf16 RM | None | Shift parameter |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| dtype | BFLOAT16 | "Input must be bfloat16" |
| layout | ROW_MAJOR | "Input must be row-major" |
| memory | INTERLEAVED (DRAM) | "Input must be interleaved" |
| rank | >= 2 | "Input must be at least 2D" |
| H alignment | divisible by 32 | "Height must be multiple of 32" |
| W alignment | divisible by 32 | "Width must be multiple of 32" |

### Output Tensor Specification
- **Shape**: same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED (DRAM)

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce is trivial single-tile |
| gamma=None, beta=None | Pure normalization, skip affine phases |
| gamma provided, beta=None | Scale only, skip add-beta phase |
| Very large W | May exceed L1; design assumes Wt tiles fit in L1 for persistence |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add gamma/beta tile reads, reduce scaler + epsilon generation |
| Compute | softmax + new | compute_core | Replace max/exp/sum with mean/sub/square/var/rsqrt using helpers |
| Writer | untilize | output_stage | Standard RM stick extraction from untilized CB |

### Work Distribution
- **Work unit**: tile-row (one horizontal row of Wt tiles = 32 RM sticks)
- **Grid**: 1D, from `split_blocks_for_tilize(grid, nblocks)` where `nblocks = H_total / 32`
- **Work per core**: `ceil(nblocks / num_cores)` tile-rows
- **Remainder**: cliff core gets `nblocks % nblocks_per_core` tile-rows

`H_total = product of all dims except last = N * C * H` (shape flattened to 2D: `[H_total, W]`).
`nblocks = H_total / 32`. Each block = 32 sticks = 1 tile-row of Wt tiles.

### Data Flow
Reader fetches 32 RM sticks per block into c_0, compute tilizes to c_24, performs multi-phase normalization in tile domain, untilizes result to c_16, writer extracts 32 RM sticks from c_16 to DRAM.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input_rm | RM sticks staging | Reader | Compute (tilize) | Wt | Block |
| c_1 | cb_gamma | Gamma tiles (optional) | Reader | Compute (mul) | Wt | Program |
| c_2 | cb_beta | Beta tiles (optional) | Reader | Compute (add) | Wt | Program |
| c_8 | cb_reduce_scaler | Reduce scaler (1/W) | Reader | Compute (reduce) | 1 | Program |
| c_10 | cb_eps | Epsilon scalar tile | Reader | Compute (add_eps) | 1 | Program |
| c_16 | cb_output_rm | Untilized output | Compute (untilize) | Writer | Wt | Block |
| c_24 | cb_tilized | Tilized input | Compute (tilize) | Compute (reduce, sub) | Wt | Row |
| c_25 | cb_mean | Mean col vector | Compute (reduce) | Compute (sub) | 2 | Row |
| c_26 | cb_centered | x - mean | Compute (sub) | Compute (square, mul) | Wt | Row |
| c_27 | cb_sq_centered | (x - mean)^2 | Compute (square) | Compute (reduce) | Wt | Row |
| c_28 | cb_var | Variance col vector | Compute (reduce) | Compute (add_eps) | 2 | Row |
| c_29 | cb_inv_std | rsqrt(var + eps) | Compute (add+rsqrt) | Compute (mul) | 2 | Row |
| c_30 | cb_pre_untilize | Final tiles before untilize | Compute (last phase) | Compute (untilize) | Wt | Row |

All tile-sized CBs use `tile_size(DataFormat::Float16_b)` = 2048 bytes per page.
Exception: c_28 (variance) uses Float32 (4096 bytes/page) for accumulation precision.

### Kernel Arguments

**Compile-time** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | RM stick size in bytes: W * 2 |
| 1 | Wt | uint32_t | Tiles per row: W / 32 |
| 2 | has_gamma | uint32_t | 1 if gamma provided, 0 otherwise |
| 3 | has_beta | uint32_t | 1 if beta provided, 0 otherwise |
| 4+ | TensorAccessorArgs (input) | uint32_t[] | Input buffer accessor |

**Compile-time** (compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Tile-rows to process (differs for cliff) |
| 1 | Wt | uint32_t | Tiles per row |
| 2 | has_gamma | uint32_t | 1 if gamma provided |
| 3 | has_beta | uint32_t | 1 if beta provided |

**Compile-time** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | RM stick size in bytes: W * 2 |
| 1 | Wt | uint32_t | Tiles per row |
| 2+ | TensorAccessorArgs (output) | uint32_t[] | Output buffer accessor |

**Runtime** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Input buffer DRAM address |
| 1 | num_rows_per_core | uint32_t | Tile-rows on this core |
| 2 | start_stick_id | uint32_t | First RM stick ID for this core |
| 3 | gamma_addr | uint32_t | Gamma buffer DRAM address (0 if none) |
| 4 | beta_addr | uint32_t | Beta buffer DRAM address (0 if none) |

**Runtime** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Output buffer DRAM address |
| 1 | num_rows_per_core | uint32_t | Tile-rows on this core |
| 2 | start_stick_id | uint32_t | First output stick ID for this core |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (Wt for tile CBs, 1 for scalar CBs)
- [x] Reduce scaler CB (c_8) is bfloat16
- [x] DEST register: 4 tiles with fp32_dest_acc_en + half-sync (limits block_size for helpers)
- [x] RM CBs count pages in tiles (tile_size pages), writer extracts sticks manually
- [x] c_28 (variance) uses Float32 for accumulation precision

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` or manual PyTorch

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
| c_0 | Wt | RM sticks (tile-sized pages) | All | Block |
| c_1 | Wt | TILE bf16 | All (gamma row) | Program |
| c_2 | Wt | TILE bf16 | All (beta row) | Program |
| c_8 | 1 | TILE bf16 | Row0 (scaler) | Program |
| c_10 | 1 | TILE bf16 | Row0 (eps scalar) | Program |
| c_16 | Wt | TILE bf16 (untilized RM data) | All | Block |
| c_24 | Wt | TILE bf16 | All | Row |
| c_25 | 2 | TILE bf16 | Col0 (reduce row output) | Row |
| c_26 | Wt | TILE bf16 | All | Row |
| c_27 | Wt | TILE bf16 | All | Row |
| c_28 | 2 | TILE fp32 | Col0 (reduce row output) | Row |
| c_29 | 2 | TILE bf16 | Col0 | Row |
| c_30 | Wt | TILE bf16 | All | Row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 3 (sub mean) | SUB | All (tilized input) | Col0 (mean from REDUCE_ROW) | COL |
| 4 (square) | SQUARE | All (centered) | same | NONE (self) |
| 6 (add eps) | ADD | Col0 (var from REDUCE_ROW) | Scalar (eps) | SCALAR |
| 7 (mul inv_std) | MUL | All (centered) | Col0 (inv_std from REDUCE_ROW) | COL |
| 8 (mul gamma) | MUL | All (normalized) | All (gamma row [1,Wt]) | NONE |
| 9 (add beta) | ADD | All (scaled) | All (beta row [1,Wt]) | NONE |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader + tilize + identity untilize + writer | `x` (passthrough) |
| 2 | reduce_mean_sub | Phases 2-3: reduce mean + subtract mean | `x - x.mean(dim=-1, keepdim=True)` |
| 3 | variance_normalize | Phases 4-7: square, reduce var, add_eps+rsqrt, multiply | Full layer norm (no affine) |
| 4 | affine_transform | Phases 8-9: multiply gamma, add beta | Full layer norm with gamma+beta |

### Stage 1: data_pipeline
- **Scope**: reader, compute (tilize + untilize only), writer
- **Reference**: `x` (identity passthrough)
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes c_0 -> c_24, then immediately untilizes c_24 -> c_16. All normalization phases skipped. Reader generates scaler/eps tiles but they are unused.

### Stage 2: reduce_mean_sub
- **Scope**: compute kernel adds phases 2-3
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: Adds reduce<SUM, REDUCE_ROW> for mean, sub<COL> for centering between tilize and untilize

### Stage 3: variance_normalize
- **Scope**: compute kernel adds phases 4-7
- **Reference**: Full layer norm without affine:
  ```python
  mean = x.mean(dim=-1, keepdim=True)
  var = ((x - mean)**2).mean(dim=-1, keepdim=True)
  (x - mean) / torch.sqrt(var + 1e-5)
  ```
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Adds square, reduce var, add_eps+rsqrt, mul phases

### Stage 4: affine_transform
- **Scope**: reader adds gamma/beta tile reads + tilize; compute adds phases 8-9
- **Reference**: `torch.nn.functional.layer_norm(x, [W], weight=gamma, bias=beta, eps=1e-5)`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: Reader reads gamma/beta tiles, compute applies mul(gamma) + add(beta)

### Reader Kernel
Reads 32 RM sticks per block into c_0 using TensorAccessor (same pattern as tilize reference). On first invocation, generates constant tiles:
- c_8: reduce scaler via `dataflow_kernel_lib::prepare_reduce_scaler<c_8>(1.0f / W)`
- c_10: epsilon tile via `dataflow_kernel_lib::prepare_reduce_scaler<c_10>(epsilon)`

If gamma/beta present: reads their RM sticks (32 sticks, full width) into c_1/c_2 once at program start. The compute kernel tilizes them from c_1/c_2 into separate intermediate CBs.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_8, c_16);`

Wait for persistent scalar CBs once:
```cpp
cb_wait_front(c_8, 1);   // reduce scaler
cb_wait_front(c_10, 1);  // epsilon
```
If gamma: `cb_wait_front(c_1, Wt);`
If beta: `cb_wait_front(c_2, Wt);`

**Per-row loop** (num_rows_per_core iterations):

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<c_0, c_24,
    tilize_config::InitAndUninit,
    tilize_config::WaitBlock>(Wt, 1);
```
- In: c_0 [Wt pages RM, pushed by reader]
- Out: c_24 [Wt tiles, tilized]

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| c_0 | 0 | freed by tilize helper |
| c_24 | Wt | freshly pushed |

#### Phase 2: Reduce Mean
```cpp
cb_wait_front(c_24, Wt);
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::NoWaitNoPop>(
    c_24, c_8, c_25,
    ReduceInputBlockShape::row(Wt));
```
- A: c_24 [Wt tiles, FRESHLY PUSHED by Phase 1, must cb_wait_front before; NoWaitNoPop means reduce won't wait/pop]
- Scaler: c_8 [1 tile, persistent, 1/W]
- Out: c_25 [1 tile, mean col vector]
- c_24: NOT popped -- persists for Phase 3

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 | Wt | waited, not popped -- persists for Phase 3 |
| c_25 | 1 | freshly pushed (mean) |

#### Phase 3: Subtract Mean
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_24, c_25, c_26,
    BinaryInputBlockShape::row(Wt));
cb_pop_front(c_24, Wt);   // free tilized input
cb_pop_front(c_25, 1);    // free mean
```
- A: c_24 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop]
- B: c_25 [1 tile, WaitUpfrontNoPop -- helper waits, keeps]
- Out: c_26 [Wt tiles, centered]
- Manual pops after: c_24, c_25

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_24 | 0 | freed |
| c_25 | 0 | freed |
| c_26 | Wt | freshly pushed (centered) |

#### Phase 4: Square Centered
```cpp
cb_wait_front(c_26, Wt);
compute_kernel_lib::square<
    BinaryInputPolicy::NoWaitNoPop>(
    c_26, c_27,
    BinaryInputBlockShape::row(Wt));
```
- A: c_26 [Wt tiles, FRESHLY PUSHED by Phase 3, must cb_wait_front; NoWaitNoPop keeps for Phase 7]
- Out: c_27 [Wt tiles, squared centered]

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| c_26 | Wt | waited, not popped -- persists for Phase 7 |
| c_27 | Wt | freshly pushed (squared) |

#### Phase 5: Reduce Variance
```cpp
cb_wait_front(c_27, Wt);
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::NoWaitNoPop>(
    c_27, c_8, c_28,
    ReduceInputBlockShape::row(Wt));
cb_pop_front(c_27, Wt);   // free squared centered
```
- A: c_27 [Wt tiles, must cb_wait_front; NoWaitNoPop]
- Scaler: c_8 [1 tile, persistent, 1/W]
- Out: c_28 [1 tile, variance col vector]
- Manual pop: c_27

**CB state after Phase 5:**
| CB | Tiles | State |
|----|-------|-------|
| c_26 | Wt | still persisting (centered) |
| c_27 | 0 | freed |
| c_28 | 1 | freshly pushed (variance) |

#### Phase 6: Add Epsilon + Rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_28, c_10, c_29,
    BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_28 [1 tile, WaitAndPopPerTile -- wait and consume variance]
- B: c_10 [1 tile, persistent epsilon, NoWaitNoPop -- already waited at startup]
- Out: c_29 [1 tile, rsqrt(var + eps)]
- Post-op: rsqrt applied in DEST before pack

#### Phase 7: Multiply by Inverse Std
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_26, c_29, c_30,
    BinaryInputBlockShape::row(Wt));
cb_pop_front(c_26, Wt);   // free centered
cb_pop_front(c_29, 1);    // free inv_std
```
- A: c_26 [Wt tiles, ALREADY WAITED from Phase 4, NoWaitNoPop]
- B: c_29 [1 tile, WaitUpfrontNoPop]
- Out: c_30 [Wt tiles, normalized]
- Manual pops after: c_26, c_29

**CB state after Phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| c_26 | 0 | freed |
| c_29 | 0 | freed |
| c_30 | Wt | freshly pushed (normalized) |

#### Phase 8: Multiply Gamma (conditional: has_gamma)
```cpp
compute_kernel_lib::mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_30, c_1, c_24,
    BinaryInputBlockShape::row(Wt));
```
- A: c_30 [Wt tiles, WaitAndPopPerTile -- consume normalized]
- B: c_1 [Wt tiles, persistent gamma, NoWaitNoPop -- already waited at startup]
- Out: c_24 [Wt tiles, scaled; reuse CB after Phase 1 freed it]

Note: When gamma is present, output goes to c_24. When gamma is absent, the untilize input CB is c_30 directly.

#### Phase 9: Add Beta (conditional: has_beta)
```cpp
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_24, c_2, c_30,
    BinaryInputBlockShape::row(Wt));
```
- A: c_24 [Wt tiles, WaitAndPopPerTile -- consume scaled]
- B: c_2 [Wt tiles, persistent beta, NoWaitNoPop -- already waited at startup]
- Out: c_30 [Wt tiles, final affine output; reuse CB]

Note: When beta is present, output goes back to c_30. Untilize input is always c_30 when affine is enabled (both gamma and beta). When only gamma (no beta), untilize input is c_24.

**CB routing summary for untilize input**:
| Config | Untilize reads from |
|--------|---------------------|
| No gamma, no beta | c_30 |
| Gamma only | c_24 |
| Gamma + beta | c_30 |

This routing is determined at compile time via `has_gamma`/`has_beta` defines.

#### Phase 10: Untilize
```cpp
// cb_untilize_in is c_30 or c_24 depending on config (compile-time)
compute_kernel_lib::untilize<Wt, cb_untilize_in, c_16,
    untilize_config::InitAndUninit,
    untilize_config::WaitBlock>(1);
```
- In: cb_untilize_in [Wt tiles, final compute result]
- Out: c_16 [Wt pages, untilized RM data]

### Writer Kernel
Waits for Wt tiles in c_16 per block. Extracts 32 RM sticks using base L1 address + row stride pattern (same as untilize reference). Each stick is `W * 2` bytes. Uses TensorAccessor for interleaved RM output writes. One `noc_async_write_barrier()` per block.

### Critical Notes
1. **cb_wait_front before NoWaitNoPop**: Every NoWaitNoPop helper call MUST be preceded by an explicit `cb_wait_front` on that CB. The wait in Phase 2 (for c_24) covers Phases 2, 3, and implicitly 4.
2. **Manual cb_pop_front after NoWaitNoPop**: After using NoWaitNoPop on A input, caller MUST pop. Documented per-phase above.
3. **Float32 CB for variance (c_28)**: Ensures accumulation precision during sum-of-squares reduction. The reduce helper handles mixed format via reconfig.
4. **Gamma/beta tilization**: Gamma/beta arrive as RM sticks. Reader writes them into c_1/c_2 as RM data. Compute tilizes c_1 -> intermediate and c_2 -> intermediate at program start. Alternative: host converts to TILE before kernel launch (simpler, recommended for initial impl).
5. **Untilize input routing**: Compile-time `constexpr` selects the correct pre-untilize CB based on `has_gamma`/`has_beta`.
6. **Epsilon via prepare_reduce_scaler**: The epsilon constant tile uses the reduce scaler format (value in row 0 of each face). This is compatible with `add<SCALAR>` which reads the tile value from position [0,0] and broadcasts.
7. **Block size for binary_op helpers**: With fp32_dest_acc_en and half-sync, DEST has 4 registers. The binary_op and reduce helpers auto-detect DEST_AUTO_LIMIT and chunk accordingly.

### Implementation Checklist
- [ ] Reader: RM stick reads (tilize pattern), generate reduce scaler + epsilon tiles, optional gamma/beta reads
- [ ] Compute: 10 phases using helpers: tilize, reduce(SUM,REDUCE_ROW), sub(COL), square, reduce(SUM,REDUCE_ROW), add(SCALAR)+rsqrt, mul(COL), optional mul(NONE), optional add(NONE), untilize
- [ ] Writer: RM stick extraction from untilized CB (untilize pattern)
- [ ] CB push/pop balance verified per-phase above
