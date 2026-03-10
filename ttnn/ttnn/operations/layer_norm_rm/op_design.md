# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operation(s)**: tilize (input_stage), batch_norm (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[b,h] = (1/W) * sum_w(input[b,h,w])
centered[b,h,w] = input[b,h,w] - mean[b,h]
var[b,h] = (1/W) * sum_w(centered[b,h,w]^2)
output[b,h,w] = centered[b,h,w] / sqrt(var[b,h] + eps)
if gamma: output *= gamma[w]
if beta:  output += beta[w]
```
Layer normalization across the width (last) dimension of row-major interleaved tensors. Reduction uses population variance (correction=0).

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No (keyword-only) | > 0 | 1e-5 | Numerical stability constant |
| gamma | Tensor | No | shape (1,1,1,W), bf16, RM | None | Scale parameter |
| beta | Tensor | No | shape (1,1,1,W), bf16, RM | None | Shift parameter |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Dtype | BFLOAT16 | "Input must be bfloat16" |
| Layout | ROW_MAJOR | "Input must be row-major" |
| Memory | Interleaved | "Input must be interleaved" |
| Rank | >= 2 | "Input rank must be >= 2" |
| Width | multiple of 32 | "Width must be tile-aligned" |
| Height | multiple of 32 | "Height must be tile-aligned" |

### Output Tensor Specification
- **Shape**: same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: interleaved

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce produces 1 tile directly |
| gamma provided, beta absent | Apply scale only |
| gamma absent, beta provided | Apply shift only |
| Large width (W >> 32) | Wt tiles per block, single-buffered |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add gamma/beta reading with 32x replication; add scaler+epsilon CB fill |
| Compute | batch_norm (adapted) | compute_core | Replace per-channel pattern with per-row REDUCE_ROW; use kernel_lib helpers instead of raw LLK |
| Writer | untilize | output_stage | Minimal changes; write RM sticks from untilize output CB |

### Work Distribution
- **Work unit**: block (1 tile-row = 32 sticks spanning full width = Wt tiles)
- **Grid**: 1D linear, up to device grid area
- **Work per core**: `nblocks_per_core = ceil(total_blocks / ncores)`, where `total_blocks = H_total / 32`
- **Remainder**: cliff core gets `total_blocks % nblocks_per_core` blocks

### Data Flow
RM sticks read from DRAM -> tilize to tile domain -> 7-phase normalization compute (all in tile domain) -> untilize back to RM sticks -> write to DRAM. Each core processes its assigned blocks independently. Per-block: reader pushes 32 sticks, compute tilizes + normalizes + untilizes, writer drains 32 sticks.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input_rm | RM sticks from reader | Reader | Compute (tilize) | Wt | Per block |
| c_5 | cb_gamma | Tilized gamma (optional) | Compute (tilize) | Compute (mul) | Wt | Program |
| c_6 | cb_beta | Tilized beta (optional) | Compute (tilize) | Compute (add) | Wt | Program |
| c_8 | cb_scaler | Reduce scaler (1/W) | Reader | Compute (reduce) | 1 | Program |
| c_9 | cb_eps | Epsilon constant tile | Reader | Compute (add) | 1 | Program |
| c_16 | cb_tilized | Multi-use: tilized input, squared, normalized, pre-untilize | Compute | Compute | Wt | Reused per phase |
| c_17 | cb_output_rm | Untilized output for writer | Compute (untilize) | Writer | Wt | Per block |
| c_24 | cb_reduce_out | Mean or variance (1 tile per block) | Compute (reduce) | Compute (sub/add) | 1 | Per phase |
| c_25 | cb_centered | Centered values; also reused for affine intermediates | Compute (sub) | Compute (square, mul) | Wt | Phases 3-7, reused 8-9 |
| c_27 | cb_rstd | rsqrt(var+eps) | Compute (add+rsqrt) | Compute (mul) | 1 | Phase 6-7 |

**CB reuse pattern per block:**
```
tilize:     c_0 -> c_16 (c_16 alive)
reduce_mean: c_16 -> c_24 (c_16 alive NoPop, c_24 alive)
sub_mean:   c_16,c_24 -> c_25 (c_16 freed, c_24 freed, c_25 alive)
square:     c_25 -> c_16 (c_25 alive NoPop, c_16 alive)
reduce_var: c_16 -> c_24 (c_16 freed, c_24 alive)
eps_rsqrt:  c_24,c_9 -> c_27 (c_24 freed, c_27 alive)
mul_rstd:   c_25,c_27 -> c_16 (c_25 freed, c_27 freed, c_16 alive)
[mul_gamma: c_16,c_5 -> c_25 (c_16 freed, c_25 alive)]
[add_beta:  c_25,c_6 -> c_16 (c_25 freed, c_16 alive)]
untilize:   final_cb -> c_17 (final_cb freed, c_17 alive for writer)
```

### Kernel Arguments

**Compile-time** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Width * sizeof(bf16) |
| 1+ | TensorAccessorArgs(input) | auto | Input buffer address mapping |

**Compile-time** (compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_blocks_per_core | uint32_t | Blocks this core processes |
| 1 | Wt | uint32_t | Tiles per row (width / 32) |
| 2 | has_gamma | uint32_t | 1 if gamma provided |
| 3 | has_beta | uint32_t | 1 if beta provided |

**Compile-time** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Width * sizeof(bf16) |
| 1 | Wt | uint32_t | Tiles per row |
| 2+ | TensorAccessorArgs(output) | auto | Output buffer address mapping |

**Runtime** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | start_stick_id | uint32_t | First stick for this core |
| 2 | num_sticks | uint32_t | Total sticks = num_blocks * 32 |
| 3 | gamma_addr | uint32_t | Gamma buffer address (0 if absent) |
| 4 | beta_addr | uint32_t | Beta buffer address (0 if absent) |

**Runtime** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | start_stick_id | uint32_t | First output stick for this core |
| 2 | num_sticks | uint32_t | Total sticks to write |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (verified per-phase)
- [x] Reduce scaler CB (c_8) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync); Wt <= 8 per chunk handled by helpers
- [x] RM CBs count pages in sticks (c_0: Wt tile-pages filled with 32 sticks), tile CBs count in tiles

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` or manual PyTorch formula
- Test shapes:

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
| c_0 | Wt | Tile-sized (holds RM sticks) | All (32 RM sticks) | Per block |
| c_5 | Wt | Tile | All (gamma replicated 32x) | Program |
| c_6 | Wt | Tile | All (beta replicated 32x) | Program |
| c_8 | 1 | Tile (bf16) | Row0 only (scaler) | Program |
| c_9 | 1 | Tile (bf16) | Row0 only (epsilon) | Program |
| c_16 | Wt | Tile | All | Reused per phase |
| c_17 | Wt | Tile-sized (holds RM sticks) | All | Per block |
| c_24 | 1 | Tile | Col0 (reduce output) | Per phase |
| c_25 | Wt | Tile | All | Phases 3-7 |
| c_27 | 1 | Tile | Col0 (rstd) | Phase 6-7 |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| sub_mean | SUB | All (c_16 tilized) | Col0 (c_24 REDUCE_ROW out) | COL |
| mul_rstd | MUL | All (c_25 centered) | Col0 (c_27 rstd) | COL |
| eps_rsqrt | ADD | Col0 (c_24 variance) | Row0 (c_9 eps constant) | SCALAR |
| mul_gamma | MUL | All (c_16 normalized) | All (c_5 gamma replicated) | NONE |
| add_beta | ADD | All (c_25 scaled) | All (c_6 beta replicated) | NONE |

Gamma/beta are replicated 32x during read, so after tilize all 32 rows are valid. Using BroadcastDim::NONE for these (not ROW) avoids the half-valid tile issue.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | tilize + untilize (identity) | input passthrough |
| 2 | subtract_mean | reduce_mean + sub_mean | x - mean(x, dim=-1) |
| 3 | normalize | square + reduce_var + eps_rsqrt + mul_rstd | layer_norm(x) |
| 4 | affine_transform | mul_gamma + add_beta | layer_norm(x, gamma, beta) |

### Stage 1: data_pipeline
- **Scope**: reader, compute (tilize+untilize only), writer
- **Reference**: `input_tensor` (identity)
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute tilizes c_0->c_16 then immediately untilizes c_16->c_17. All normalization phases skipped.

### Stage 2: subtract_mean
- **Scope**: compute adds reduce_mean + sub_mean phases between tilize and untilize
- **Reference**: `input_tensor - input_tensor.mean(dim=-1, keepdim=True)`
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from stage 1**: +2 compute phases (reduce, sub). Reader adds scaler CB fill. Untilize source changes from c_16 to c_25 (centered output).

### Stage 3: normalize
- **Scope**: compute adds square + reduce_var + eps_rsqrt + mul_rstd
- **Reference**: `torch.nn.functional.layer_norm(input_tensor, [input_tensor.shape[-1]], eps=1e-5)`
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from stage 2**: +4 compute phases. Reader adds epsilon CB fill. Untilize source changes from c_25 back to c_16 (normalized output). c_25 now persists through mul_rstd instead of being untilized.

### Stage 4: affine_transform
- **Scope**: compute adds mul_gamma + add_beta; reader adds gamma/beta reading+replication+tilize
- **Reference**: `torch.nn.functional.layer_norm(input_tensor, [input_tensor.shape[-1]], weight=gamma, bias=beta, eps=1e-5)`
- **Shapes**: (1,1,32,32), (1,1,64,128), (1,1,32,256), (4,2,64,64)
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from stage 3**: +2 compute phases (mul, add), reader gamma/beta logic, gamma/beta tilize in compute init

### Reader Kernel
Reads RM sticks from interleaved DRAM using TensorAccessor. Per block: `cb_reserve_back(c_0, Wt)` -> 32x `noc_async_read` (one per stick) -> `noc_async_read_barrier` -> `cb_push_back(c_0, Wt)`. Before main loop: fills c_8 with reduce scaler via `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<c_8, SUM, REDUCE_ROW, 32, W>()` and c_9 with epsilon via `dataflow_kernel_lib::prepare_reduce_scaler<c_9>(epsilon)`. For gamma/beta: reads single W-byte stick, replicates 32x into temporary CB, signals compute to tilize.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_8, c_16)`

Per-block loop (num_blocks iterations):

#### Phase 1: Tilize (RM -> tiles)
```cpp
compute_kernel_lib::tilize<c_0, c_16, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(Wt, 1);
```
- In: c_0 [Wt pages, reader-pushed RM sticks]
- Out: c_16 [Wt tiles, freshly pushed]

#### Phase 2: Reduce Mean
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop, INPUT_AND_OUTPUT>(
    c_16, c_8, c_24, ReduceInputBlockShape::row(Wt));
```
- In: c_16 [Wt tiles, FRESHLY PUSHED by Phase 1, WaitUpfront waits then holds]
- Scaler: c_8 [1 tile, persistent, waited internally by reduce]
- Out: c_24 [1 tile, pushed]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_16 | Wt | waited, not popped — persists for Phase 3 |
| c_24 | 1 | freshly pushed (mean) |

#### Phase 3: Subtract Mean
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_16, c_24, c_25, BinaryInputBlockShape::row(Wt));
```
- A: c_16 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitPopAtEnd — popped at end]
- B: c_24 [1 tile, WaitUpfrontPopAtEnd — waited then popped at end]
- Out: c_25 [Wt tiles, pushed per-tile]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| c_16 | 0 | freed (popped at end by NoWaitPopAtEnd) |
| c_24 | 0 | freed (popped at end by WaitUpfrontPopAtEnd) |
| c_25 | Wt | freshly pushed (centered) |

#### Phase 4: Square Centered Values
```cpp
compute_kernel_lib::square<BinaryInputPolicy::WaitUpfrontNoPop>(
    c_25, c_16, BinaryInputBlockShape::row(Wt));
```
- In: c_25 [Wt tiles, WaitUpfrontNoPop — waited then held for Phase 7]
- Out: c_16 [Wt tiles, pushed per-tile]

#### Phase 5: Reduce Variance
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, BulkWaitBulkPop, INPUT_AND_OUTPUT>(
    c_16, c_8, c_24, ReduceInputBlockShape::row(Wt));
```
- In: c_16 [Wt tiles, BulkWaitBulkPop — waited Wt, processed, popped Wt]
- Scaler: c_8 [1 tile, persistent]
- Out: c_24 [1 tile, pushed (variance)]

#### Phase 6: Epsilon + Rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_24, c_9, c_27, BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) { rsqrt_tile_init(); rsqrt_tile(dst_idx); });
```
- A: c_24 [1 tile, WaitAndPopPerTile — waited, processed, popped]
- B: c_9 [1 tile, persistent epsilon, WaitUpfrontNoPop — waited (already present), never popped]
- Out: c_27 [1 tile, pushed (rstd = 1/sqrt(var+eps))]
- Post-op: rsqrt applied in DEST before pack

#### Phase 7: Multiply by Rstd (Normalize)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitUpfrontPopAtEnd>(
    c_25, c_27, c_16, BinaryInputBlockShape::row(Wt));
```
- A: c_25 [Wt tiles, ALREADY WAITED from Phase 4, NoWaitPopAtEnd — popped at end]
- B: c_27 [1 tile, WaitUpfrontPopAtEnd — waited, popped at end]
- Out: c_16 [Wt tiles, pushed (normalized)]

After Phase 7, `final_cb = c_16`.

#### Phase 8 (optional): Multiply Gamma
```cpp
compute_kernel_lib::mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_16, c_5, c_25, BinaryInputBlockShape::row(Wt));
```
- A: c_16 [Wt tiles, streaming pop]
- B: c_5 [Wt tiles, persistent gamma, never popped]
- Out: c_25 [Wt tiles]

After Phase 8, `final_cb = c_25`.

#### Phase 9 (optional): Add Beta
```cpp
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    c_25, c_6, c_16, BinaryInputBlockShape::row(Wt));
```
- A: c_25 [Wt tiles, streaming pop]
- B: c_6 [Wt tiles, persistent beta, never popped]
- Out: c_16 [Wt tiles]

After Phase 9, `final_cb = c_16`.

#### Phase 10: Untilize (tiles -> RM)
```cpp
compute_kernel_lib::untilize<Wt, final_cb, c_17, InitAndUninit, WaitBlock, UnpackAndPackReconfigure>(1);
```
- In: final_cb [Wt tiles, WaitBlock waits for Wt tiles]
- Out: c_17 [Wt tile-pages of RM data, pushed]

`final_cb` is determined at compile time: c_16 (no affine or gamma+beta), c_25 (gamma only or beta only).

### Writer Kernel
Reads untilized RM sticks from c_17. Per block: `cb_wait_front(c_17, Wt)` -> extract 32 sticks from tile-pages (each stick at offset `j * stick_size` from CB read ptr) -> `noc_async_write` each stick to output DRAM via TensorAccessor -> `noc_async_write_barrier` -> `cb_pop_front(c_17, Wt)`.

### Critical Notes
1. **c_8 scaler uses `calculate_and_prepare_reduce_scaler<c_8, SUM, REDUCE_ROW, 32, W>()`** — the W template arg is the reduce factor. For layer norm, scaler = 1/W since we use SUM reduce with AVG-equivalent scaling. Actually, use `prepare_reduce_scaler<c_8>(1.0f / W)` since W is a runtime value. The `calculate_and_prepare_reduce_scaler` requires compile-time reduce_factor.
2. **Epsilon CB uses `prepare_reduce_scaler<c_9>(epsilon)`** — fills row 0 of the tile with the epsilon value in bfloat16 packed format. The SCALAR broadcast in Phase 6 uses only element [0,0].
3. **NoWaitPopAtEnd on A after persistent NoPop** — When Phase 2 uses WaitUpfrontNoPop on c_16, the tiles remain. Phase 3's NoWaitPopAtEnd skips the wait (tiles already visible) and pops at the end. This is the correct pattern for reusing tiles across operations.
4. **Gamma/beta tilize at program start** — Reader writes gamma/beta sticks 32x into a temp CB (c_0 reused before main loop), compute tilizes into c_5/c_6. These persist for all blocks.
5. **`rsqrt_tile_init()` in Phase 6 post-op** — Must be called before `rsqrt_tile()` inside the lambda. The init sets up SFPU configuration for rsqrt.

### Implementation Checklist
- [ ] Reader: TensorAccessor for RM sticks, scaler/epsilon fill, optional gamma/beta replication
- [ ] Compute: 7-10 phases using helpers: tilize, reduce (x2), sub, square, add+rsqrt, mul, [mul, add], untilize
- [ ] Writer: TensorAccessor for RM output sticks, 32-stick extraction per block
- [ ] CB push/pop balance verified (reuse pattern above)
