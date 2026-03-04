# Operation Design: layer_norm

## Overview
- **Operation Name**: layer_norm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**:
  - `moreh_norm_w` (compute_core): Width reduction with cross-tile accumulation + intra-tile reduce
  - `softmax_general` (compute_patterns): Multi-pass normalization (3 passes over input for large variant)

## Mathematical Definition
```
E[x]   = (1/W) * sum(x[n, 0..W-1])              -- mean across width
Var[x] = (1/W) * sum((x[n, i] - E[x])^2)        -- variance across width
y[n,i] = gamma[i] * (x[n,i] - E[x]) / sqrt(Var[x] + eps) + beta[i]
```
LayerNorm normalizes each row independently across the width (last) dimension.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| eps | float | No | > 0 | 1e-5 | Variance stability constant |
| gamma | Tensor | No | shape [1,W] or [1,1,1,W] | None | Per-element scale |
| beta | Tensor | No | shape [1,W] or [1,1,1,W] | None | Per-element shift |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Layout | TILE_LAYOUT | Must be tilized |
| Dtype | BFLOAT16 | Only bf16 supported |
| Memory | Interleaved DRAM | No sharding |
| Rank | 2D [N,W] or 4D [1,1,N,W] | Internally treated as [N,W] |
| Width | Multiple of 32 (TILE_WIDTH) | Pad if needed before dispatch |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: BFLOAT16
- **Layout**: TILE_LAYOUT
- **Memory**: Interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| gamma=None, beta=None | Skip mul/add in pass 3 |
| gamma only | Apply scale, skip bias |
| beta only | Skip scale, apply bias |
| W = 32 (single tile) | Wt=1, cross-tile accumulation loop is trivial |
| N = 1, W = 32 | Minimal case: 1 core, 1 row, 1 tile |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | softmax_w_large | input_stage | 3-pass read pattern; add gamma/beta reads in pass 3 |
| Compute (reduce) | moreh_norm_w | compute_core | Cross-tile add + reduce_row for mean/variance |
| Compute (normalize) | softmax_w_large | compute_core | sub_bcast + mul_bcast pattern for normalization |
| Writer | moreh_norm_w | output_stage | Standard tile-per-tile write, same shape as input |

### Work Distribution
- **Work unit**: tile-row (Wt tiles sharing same row index)
- **Grid**: Up to `compute_with_storage_grid_size()`, capped at num_tile_rows
- **Work per core**: `ceil(N*Ht / num_cores)` tile-rows (two-group split)
- **Remainder**: `split_work_to_cores()` with core_group_1 / core_group_2

### Data Flow

Three-pass large variant. Each tile-row is read 3 times from DRAM. Pass 1 computes mean, pass 2 computes variance, pass 3 produces normalized output. Single-buffered CBs (capacity=1) for simplicity.

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_input | Input tile (streaming) | Reader | Compute | 1 | Block (per tile) |
| c_1 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | Program (filled once) |
| c_2 | cb_eps | Epsilon tile | Reader | Compute | 1 | Program (filled once) |
| c_3 | cb_gamma | Gamma tile (streaming) | Reader | Compute | 1 | Block (per tile, pass 3 only) |
| c_4 | cb_beta | Beta tile (streaming) | Reader | Compute | 1 | Block (per tile, pass 3 only) |
| c_16 | cb_output | Output tile (streaming) | Compute | Writer | 1 | Block (per tile) |
| c_24 | cb_mean | Mean result (per-row) | Compute | Compute | 1 | Row (persists across passes) |
| c_25 | cb_accum | Cross-tile accumulator | Compute | Compute | 1 | Row (within pass) |
| c_26 | cb_var | Variance + eps result | Compute | Compute | 1 | Row (persists across passes) |
| c_27 | cb_tmp | Scratch (x-mean, etc.) | Compute | Compute | 1 | Block (per tile) |

All CBs use BFLOAT16 data format (tile_size = 2048 bytes). All are single-buffered with 1 page.

### Kernel Arguments

**Compile-time** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0+ | TensorAccessorArgs (input) | uint32_t[] | Input buffer accessor |
| Writer | 0+ | TensorAccessorArgs (output) | uint32_t[] | Output buffer accessor |
| Compute | 0 | num_rows_per_core | uint32_t | Tile-rows for this core |
| Compute | 1 | Wt | uint32_t | Width tiles per row |
| Compute | 2 | has_gamma | uint32_t | 1 if gamma present |
| Compute | 3 | has_beta | uint32_t | 1 if beta present |

**Runtime** (per kernel):

| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | input_addr | uint32_t | Input buffer address |
| Reader | 1 | num_rows_per_core | uint32_t | Tile-rows for this core |
| Reader | 2 | Wt | uint32_t | Width in tiles |
| Reader | 3 | tile_offset | uint32_t | Starting tile index |
| Reader | 4 | gamma_addr | uint32_t | Gamma buffer address (0 if none) |
| Reader | 5 | beta_addr | uint32_t | Beta buffer address (0 if none) |
| Reader | 6 | eps_bits | uint32_t | Epsilon as bfloat16 bits |
| Writer | 0 | output_addr | uint32_t | Output buffer address |
| Writer | 1 | num_rows_per_core | uint32_t | Tile-rows for this core |
| Writer | 2 | Wt | uint32_t | Width in tiles |
| Writer | 3 | tile_offset | uint32_t | Starting tile index |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (all use 1)
- [x] Reduce scaler CB (c_1) is bfloat16
- [x] DEST register holds max 8 tiles (bf16) -- all ops work on 1 tile at a time
- [x] All tile CBs count pages in tiles (1 page = 1 tile = 2048 bytes bf16)

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile | Tile iteration | `(1, 1, 64, 128)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |
| Wide | Many W tiles | `(1, 1, 32, 512)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (input) | 1 | Tile | All | Block |
| c_1 (scaler) | 1 | Tile | Row0 only | Program |
| c_2 (eps) | 1 | Tile | All (broadcast tile) | Program |
| c_3 (gamma) | 1 | Tile | All | Block |
| c_4 (beta) | 1 | Tile | All | Block |
| c_16 (output) | 1 | Tile | All | Block |
| c_24 (mean) | 1 | Tile | Col0 (REDUCE_ROW output) | Row |
| c_25 (accum) | 1 | Tile | All | Row |
| c_26 (var) | 1 | Tile | Col0 (REDUCE_ROW output) | Row |
| c_27 (tmp) | 1 | Tile | All | Block |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| Pass 2: x - mean | SUB | All (c_0) | Col0 (c_24 mean) | COL |
| Pass 2: square | SQUARE | All (c_27) | N/A | N/A |
| Pass 2: add_eps | ADD | Col0 (c_26 var) | All (c_2 eps) | SCALAR |
| Pass 3: x - mean | SUB | All (c_0) | Col0 (c_24 mean) | COL |
| Pass 3: mul rsqrt(var+eps) | MUL | All (c_27) | Col0 (c_26) | COL |
| Pass 3: mul gamma | MUL | All (c_27) | All (c_3 gamma) | NONE |
| Pass 3: add beta | ADD | All (c_27) | All (c_4 beta) | NONE |

**Note on REDUCE_ROW -> COL broadcast**: REDUCE_ROW collapses width into Col0. The COL broadcast replicates Col0 across all columns. This is the correct pairing.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader+Writer passthrough, compute identity | `x` (input passthrough) |
| 2 | mean_subtract | Pass 1 (mean) + Pass 2 (x - mean), skip pass 3 normalize | `x - x.mean(dim=-1, keepdim=True)` |
| 3 | variance | Pass 2 adds square + accum + reduce to get variance | `(x - x.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1, keepdim=True)` |
| 4 | full_normalize | Pass 3: rsqrt, multiply, gamma, beta | `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma, bias=beta, eps=1e-5)` |

### Stage 1: data_pipeline
- **Scope**: All 3 kernel files. Reader reads Wt tiles for pass 1, compute copies to output, writer writes Wt tiles. Passes 2/3 read but discard.
- **Reference**: `x` (identity passthrough)
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute copies c_0 directly to c_16 in pass 1 loop. Pass 2/3 loops just pop tiles from c_0 without processing. Writer writes from c_16.
- **Implementation notes**: Reader does 3 full loops over Wt tiles. Pass 1: push to c_0, compute copies to c_16. Pass 2 & 3: push to c_0, compute pops and discards. This validates the 3-pass data pipeline and tile-level synchronization.

### Stage 2: mean_subtract
- **Scope**: Compute kernel. Adds Pass 1 mean computation (cross-tile add_tiles into c_25, reduce_row to c_24). Pass 2 now does (x - mean) and outputs to c_16. Pass 3 still does passthrough.
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Delta from previous**: Compute adds cross-tile SUM accumulation, reduce<SUM, REDUCE_ROW> for mean (with scaler 1/W), sub<COL> for (x-mean)
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`
- **Tolerances**: rtol=0.02, atol=0.1

### Stage 3: variance
- **Scope**: Compute kernel. Pass 2 now computes (x-mean)^2, accumulates, reduces to get variance. Adds eps, rsqrt. Stores in c_26. Pass 3 still does identity.
- **Reference**: `(x - x.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1, keepdim=True)`
- **Delta from previous**: After sub<COL>, adds square(), cross-tile add_tiles for variance accumulation, reduce<SUM, REDUCE_ROW> with 1/W scaler
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Output shape**: Same as input but reduced: `(batch dims..., Ht*32, 32)` -- actually we verify the variance tile, so we use a custom reference expression

### Stage 4: full_normalize
- **Scope**: Compute kernel + reader (gamma/beta reads). Pass 3: (x - mean) * rsqrt(var + eps) * gamma + beta.
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma, bias=beta, eps=1e-5)`
- **Delta from previous**: Pass 2 adds add_eps + rsqrt to c_26. Pass 3: sub<COL>(x, mean), mul<COL>(tmp, rsqrt_var), mul<NONE>(tmp, gamma), add<NONE>(tmp, beta). Reader adds gamma/beta reads in pass 3.
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`, `(1, 1, 32, 512)`
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel

Three-pass pattern (following softmax_w_large):
1. Fill c_1 (scaler with 1/W) and c_2 (epsilon tile) once at startup using `prepare_reduce_scaler`
2. **Pass 1**: Loop Wt tiles, read input tile -> push to c_0
3. **Pass 2**: Loop Wt tiles, re-read same input tiles -> push to c_0
4. **Pass 3**: Loop Wt tiles, re-read input tiles -> push to c_0. Also read gamma tile -> push c_3, beta tile -> push c_4 (if present, one per W tile)

Uses TensorAccessor for both input and gamma/beta. Gamma/beta indexed by `col_idx` (same tile for every row).

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);`

#### Pass 1: Compute Mean
For each tile-row:
```
for col in 0..Wt:
    // Cross-tile SUM accumulation
    if col == 0: copy c_0 -> c_25
    else: add_tiles(c_0, c_25) -> c_25  // binary_op ADD
    pop c_0

// Intra-tile reduce_row to get mean
reduce<SUM, REDUCE_ROW>(c_25, c_1, c_24, ReduceInputBlockShape::single())
// c_24 now holds mean (scaler already includes 1/W factor)
```

**Helper**: `compute_kernel_lib::reduce<SUM, REDUCE_ROW>` with `ReduceInputBlockShape::single()` for the final intra-tile reduce. The cross-tile accumulation is manual (add_tiles loop). The scaler CB (c_1) contains 1/W so the reduce produces mean directly.

#### Pass 2: Compute Variance
For each tile-row:
```
for col in 0..Wt:
    // (x - mean)
    sub<COL>(c_0, c_24, c_27)  // broadcast mean across cols
    // (x - mean)^2
    square(c_27, c_27)
    // cross-tile accumulation
    if col == 0: copy c_27 -> c_25
    else: add_tiles(c_27, c_25) -> c_25
    pop c_0

// Intra-tile reduce_row for variance
reduce<SUM, REDUCE_ROW>(c_25, c_1, c_26, ReduceInputBlockShape::single())
// c_26 = variance (1/W factor in scaler)

// add eps: c_26 = c_26 + eps
add<SCALAR>(c_26, c_2, c_26)  // broadcast eps scalar

// rsqrt: c_26 = 1/sqrt(var + eps)
// Applied as post_reduce_op or separate unary
```

**Helpers used**:
- `compute_kernel_lib::sub<COL>()` for (x - mean) broadcast subtraction
- `compute_kernel_lib::square()` for squaring
- `compute_kernel_lib::reduce<SUM, REDUCE_ROW>()` for intra-tile reduce
- `compute_kernel_lib::add<SCALAR>()` for adding epsilon

**rsqrt**: After add_eps, apply `rsqrt_tile()` (raw SFPU call -- no helper wraps this). Must do: `tile_regs_acquire()`, `cb_wait_front(c_26, 1)`, unpack, `rsqrt_tile_init()`, `rsqrt_tile(0)`, pack to c_26, `tile_regs_release()`.

#### Pass 3: Normalize + Scale + Shift
For each tile-row, for each col in 0..Wt:
```
    // (x - mean)
    sub<COL>(c_0, c_24, c_27)
    // * rsqrt(var + eps)
    mul<COL>(c_27, c_26, c_27)
    // * gamma (if present)
    if has_gamma: mul<NONE>(c_27, c_3, c_27)  // gamma is full [1, Wt] tile
    // + beta (if present)
    if has_beta: add<NONE>(c_27, c_4, c_27)   // beta is full [1, Wt] tile
    // output
    copy c_27 -> c_16
    pop c_0
```

**Helpers used**:
- `compute_kernel_lib::sub<COL>()` -- (x - mean)
- `compute_kernel_lib::mul<COL>()` -- multiply by rsqrt(var+eps) (COL broadcast from reduced tile)
- `compute_kernel_lib::mul()` -- multiply by gamma (NONE broadcast, element-wise)
- `compute_kernel_lib::add()` -- add beta (NONE broadcast, element-wise)

### Writer Kernel

Standard tile streaming writer. For each tile-row, waits for Wt output tiles from c_16, writes each via TensorAccessor.

```
for row in 0..num_rows_per_core:
    for col in 0..Wt:
        tile_idx = tile_offset + row * Wt + col
        cb_wait_front(c_16, 1)
        noc_async_write_tile(tile_idx, accessor, get_read_ptr(c_16))
        noc_async_write_barrier()
        cb_pop_front(c_16, 1)
```

### Critical Notes

1. **Cross-tile accumulation is manual**: The reduce helper only handles intra-tile reduction. The outer loop that accumulates across Wt tiles via add_tiles must be written explicitly, using the c_25 ping-pong pattern from moreh_norm_w.

2. **Scaler encodes 1/W**: The scaler tile in c_1 is filled with `1.0 / W` (where W is the full unpadded width in elements). This means `reduce<SUM, REDUCE_ROW>` produces `sum * (1/W) = mean` directly. The same scaler is reused for both mean and variance reduction.

3. **c_24 (mean) and c_26 (var) persist across passes**: These CBs are written in passes 1 and 2 respectively, and read (without pop) in passes 2 and 3. The compute kernel must NOT pop these CBs until the row is fully processed. Use `NoWaitNoPop` or `WaitUpfrontNoPop` input policy when reading them.

4. **rsqrt has no helper**: Must use raw `rsqrt_tile_init()` + `rsqrt_tile(dst_idx)` calls with manual DEST register management.

5. **Gamma/beta are row-broadcast**: Each gamma/beta tile corresponds to a width position and is the same for every row. The reader reads them fresh for each row in pass 3 (they are small so this is fine).

6. **Binary op reconfig**: Between different binary ops (sub, mul, add, square) and reduce, data format reconfiguration is needed. Use `INPUT_AND_OUTPUT` reconfig mode (default) for safety between different op types.

### Implementation Checklist
- [ ] Reader: 3-pass tile streaming, scaler (1/W) fill, epsilon fill, gamma/beta reads
- [ ] Compute: Pass 1 (mean via cross-tile add + reduce_row), Pass 2 (variance via sub+square+add+reduce_row+eps+rsqrt), Pass 3 (normalize via sub+mul+gamma+beta)
- [ ] Writer: Standard tile-at-a-time write
- [ ] CB push/pop balance verified: c_0 pushed/popped Wt times per pass (3*Wt per row), c_16 pushed/popped Wt times per row (pass 3), c_24/c_26 pushed once per row, read multiple times, popped once at end of row
