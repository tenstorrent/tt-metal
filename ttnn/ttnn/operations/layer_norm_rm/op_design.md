# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**:
  - `tilize_multi_core_interleaved` (input_stage) -- reader pattern, tilize CB layout
  - `untilize_multi_core` (output_stage) -- writer pattern, untilize CB layout
  - `softmax_w_small` (compute_core) -- multi-pass row reduction, CB reuse, scalar broadcasting

## Mathematical Definition
```
For each row (last dimension) of the input tensor:
  mean = sum(x) / W
  x_centered = x - mean
  var = sum(x_centered^2) / W
  x_norm = x_centered * rsqrt(var + epsilon)
  output = gamma * x_norm + beta
```
Layer normalization along the W dimension with learnable affine parameters gamma and beta.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant added to variance before rsqrt |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Layout | ROW_MAJOR | "Input must be row-major" |
| Memory | INTERLEAVED (DRAM) | "Input must be interleaved" |
| Dtype | BFLOAT16 | "Currently only bfloat16 supported" |
| Rank | >= 2 | "Input must be at least 2D" |
| Last dim | Multiple of 32 | "Width must be tile-aligned" |
| Second-last dim | Multiple of 32 | "Height must be tile-aligned" |

### Gamma/Beta Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Shape | (1, 1, 1, W) where W matches input width |
| Layout | ROW_MAJOR |
| Memory | INTERLEAVED (DRAM) |
| Dtype | BFLOAT16 |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: BFLOAT16 (same as input)
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED (DRAM)

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Single tile per row, no multi-tile reduction loops |
| Large batch dims | Work distribution across cores handles it |
| epsilon very small | rsqrt may overflow -- user's responsibility |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize_multi_core_interleaved | input_stage | Add gamma/beta reading; read RM sticks for input, gamma, beta |
| Compute | softmax_w_small + NEW | compute_core | Replace max/exp/sum with mean/center/var/rsqrt/affine; reuse multi-pass pattern |
| Writer | untilize_multi_core | output_stage | Write untilized RM sticks to DRAM; simplified (interleaved only) |

### Work Distribution
- **Work unit**: Tile-row (one horizontal strip of 32 rows spanning the full tensor width = Wt tiles)
- **Grid**: 1D linearized from 2D compute grid
- **Work per core**: `ceil(num_tile_rows / num_cores)` tile-rows
- **Remainder**: Last core ("cliff") gets `num_tile_rows % nblocks_per_core` tile-rows
- **Tile-row count**: `(N * C * H) / 32` where the tensor is flattened to `[outer_dims, H, W]`

Each core processes its assigned tile-rows sequentially. Per tile-row, the reader loads 32 RM sticks into the input CB, loads Wt gamma tiles and Wt beta tiles (or reads them once if they fit), compute tilizes + normalizes + untilizes, and writer drains RM sticks.

### Data Flow

Per tile-row (32 rows, Wt tiles wide):

1. **Reader**: Read 32 RM sticks from DRAM into `cb_in` (c_0). Read Wt gamma tiles and Wt beta tiles into `cb_gamma`/`cb_beta`. (Gamma/beta are read once per core since they repeat across rows.)
2. **Compute**: Tilize `cb_in` -> `cb_tilized`. Then multi-pass layer norm on tilized data. Finally untilize result -> `cb_untilized`.
3. **Writer**: Drain `cb_untilized` as RM sticks to DRAM.

### Circular Buffer Requirements

| CB ID | Name | Purpose | Producer | Consumer | Pages | Page Size | Lifetime |
|-------|------|---------|----------|----------|-------|-----------|----------|
| c_0 | cb_in | RM sticks from reader (input) | Reader | Compute (tilize) | Wt | tile_size | Block (per tile-row) |
| c_1 | cb_tilized | Tilized input tiles | Compute (tilize) | Compute (reduce/sub) | Wt | tile_size | Row (persists phases 1-2) |
| c_2 | cb_scaler | Reduce scaler tile (1/W) | Reader (once) | Compute | 1 | tile_size | Program |
| c_3 | cb_eps | Epsilon scaler tile | Reader (once) | Compute | 1 | tile_size | Program |
| c_4 | cb_gamma | Gamma tiles (tilized) | Reader/Compute | Compute | Wt | tile_size | Row |
| c_5 | cb_beta | Beta tiles (tilized) | Reader/Compute | Compute | Wt | tile_size | Row |
| c_16 | cb_out | Final output tiles (untilized) | Compute (untilize) | Writer | Wt | tile_size | Block (per tile-row) |
| c_24 | cb_mean | Row-wise mean (1 tile) | Compute | Compute | 1 | tile_size | Row |
| c_25 | cb_centered | x - mean intermediate | Compute | Compute | Wt | tile_size | Row (persists phases 3-5) |
| c_26 | cb_var | Variance scalar (after rsqrt) | Compute | Compute | 1 | tile_size | Row |
| c_27 | cb_normed | Normalized output before affine | Compute | Compute | Wt | tile_size | Row |
| c_28 | cb_affine_out | After gamma*x_norm, before +beta | Compute | Compute (untilize) | Wt | tile_size | Row |

**CB c_0 sizing note**: Page size is `tile_size` (not stick size). Reader batches 32 sticks into `Wt` tile-sized pages. This matches the tilize pattern: 32 sticks of full width = Wt tiles of data. Compute waits for Wt pages in c_0.

**Gamma/Beta**: Reader reads gamma/beta as RM sticks into temporary CBs, then compute tilizes them. Since gamma/beta have shape `(1,1,1,W)`, that's only 1 stick of width W. The reader reads 32 sticks (padding the remaining 31 with the same values or zeros) to form a tilizable block. Alternatively, gamma/beta can be pre-tilized on host. For simplicity: gamma/beta are read as 32 repeated sticks and tilized by compute into cb_gamma/cb_beta. These are read ONCE per core (they don't change across rows).

**Revised gamma/beta approach**: To avoid complexity of tilizing gamma/beta in the compute kernel, the reader will read gamma/beta sticks into separate CBs (`cb_gamma_rm`, `cb_beta_rm`), and compute will tilize them using the tilize helper. Since gamma/beta shape is (1,1,1,W), we need 32 copies of the same stick to form a proper tile-row. The reader repeats the single gamma/beta stick 32 times into the CB.

**Final simplified gamma/beta approach**: Use separate RM input CBs for gamma (c_6) and beta (c_7). Reader loads 32 copies of the gamma stick into c_6 and 32 copies of the beta stick into c_7. Compute tilizes c_6 -> c_4 (cb_gamma) and c_7 -> c_5 (cb_beta). This happens ONCE per core.

Updated CB table with gamma/beta RM staging:

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| c_0 | cb_in_rm | RM sticks from reader (input) | Reader | Compute (tilize) | Wt | Block |
| c_1 | cb_tilized | Tilized input tiles | Compute | Compute | Wt | Row |
| c_2 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | Program |
| c_3 | cb_eps | Epsilon tile | Reader | Compute | 1 | Program |
| c_4 | cb_gamma | Tilized gamma | Compute | Compute | Wt | Program |
| c_5 | cb_beta | Tilized beta | Compute | Compute | Wt | Program |
| c_6 | cb_gamma_rm | RM gamma sticks (32 repeated) | Reader | Compute | Wt | Once |
| c_7 | cb_beta_rm | RM beta sticks (32 repeated) | Reader | Compute | Wt | Once |
| c_16 | cb_out | Untilized output for writer | Compute | Writer | Wt | Block |
| c_24 | cb_mean | Row-wise mean reduction output | Compute | Compute | 1 | Row |
| c_25 | cb_centered | x - mean | Compute | Compute | Wt | Row |
| c_26 | cb_inv_std | rsqrt(var + eps) | Compute | Compute | 1 | Row |
| c_27 | cb_normed | x_norm = centered * inv_std | Compute | Compute | Wt | Row |
| c_28 | cb_affine | gamma * x_norm (before + beta) | Compute | Compute | Wt | Row |

All page sizes = `tile_size` (bf16: 2048 bytes for 32x32 tile).

### Kernel Arguments

**Compile-time (Reader)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Width * element_size bytes (one RM stick) |
| 1 | gamma_stick_size | uint32_t | Same as stick_size (gamma has same width) |
| 2+ | TensorAccessorArgs (input) | uint32_t[] | Input tensor accessor |
| N+ | TensorAccessorArgs (gamma) | uint32_t[] | Gamma tensor accessor |
| M+ | TensorAccessorArgs (beta) | uint32_t[] | Beta tensor accessor |

**Compile-time (Compute)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_rows_per_core | uint32_t | Tile-rows this core processes |
| 1 | Wt | uint32_t | Tiles per row (W / 32) |

**Compile-time (Writer)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | Output stick size in bytes |
| 1+ | TensorAccessorArgs (output) | uint32_t[] | Output tensor accessor |

**Runtime (Reader)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | gamma_addr | uint32_t | Gamma buffer base address |
| 2 | beta_addr | uint32_t | Beta buffer base address |
| 3 | num_rows_of_sticks | uint32_t | Total sticks = num_tile_rows * 32 |
| 4 | start_stick_id | uint32_t | First stick ID for this core |
| 5 | scaler_value | uint32_t | Bit-cast 1/W float |
| 6 | eps_value | uint32_t | Bit-cast epsilon float |

**Runtime (Writer)**:
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |
| 1 | num_sticks | uint32_t | Total output sticks this core writes |
| 2 | start_stick_id | uint32_t | First output stick ID for this core |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB (c_2) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) / 4 tiles (f32)
- [x] RM CBs count pages in tiles (page_size = tile_size, reader batches 32 sticks per Wt push)
- [x] Epsilon CB (c_3) is bfloat16

### Test Criteria
- Output shape matches input shape exactly
- Numerical accuracy vs `torch.nn.functional.layer_norm` with weight=gamma, bias=beta
- rtol/atol per stage (see Part 2)

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile-row, single tile wide | `(1, 1, 32, 32)` |
| Multi-tile W | Tests tile iteration in reductions | `(1, 1, 32, 64)` |
| Multi-tile HW | Multiple tile-rows | `(1, 1, 64, 128)` |
| Non-square | W != H | `(1, 1, 32, 256)` |
| Multi-batch | Tests batch handling | `(2, 1, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | ID | Pages | Layout | Valid Region | Lifetime |
|----|----|-------|--------|--------------|----------|
| cb_in_rm | c_0 | Wt | RM sticks packed as tile pages | All | Block |
| cb_tilized | c_1 | Wt | TILE | All (2D) | Row |
| cb_scaler | c_2 | 1 | TILE | Scaler format | Program |
| cb_eps | c_3 | 1 | TILE | Scaler format | Program |
| cb_gamma | c_4 | Wt | TILE | All (2D) | Program |
| cb_beta | c_5 | Wt | TILE | All (2D) | Program |
| cb_gamma_rm | c_6 | Wt | RM sticks packed as tile pages | All | Once |
| cb_beta_rm | c_7 | Wt | RM sticks packed as tile pages | All | Once |
| cb_out | c_16 | Wt | RM sticks packed as tile pages | All | Block |
| cb_mean | c_24 | 1 | TILE | Col0 (REDUCE_ROW output) | Row |
| cb_centered | c_25 | Wt | TILE | All (2D) | Row |
| cb_inv_std | c_26 | 1 | TILE | Col0 (REDUCE_ROW output) | Row |
| cb_normed | c_27 | Wt | TILE | All (2D) | Row |
| cb_affine | c_28 | Wt | TILE | All (2D) | Row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| 2 (sub mean) | SUB | All (cb_tilized) | Col0 (cb_mean) | COL |
| 3 (square) | SQUARE | All (cb_centered) | N/A | NONE |
| 5 (mul inv_std) | MUL | All (cb_centered) | Col0 (cb_inv_std) | COL |
| 6 (mul gamma) | MUL | All (cb_normed) | All (cb_gamma) | NONE |
| 7 (add beta) | ADD | All (cb_affine) | All (cb_beta) | NONE |

Broadcast correctness: REDUCE_ROW outputs have Col0 valid region. Using COL broadcast correctly maps column vector across all columns.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader (RM sticks) + Compute (tilize+untilize identity) + Writer (RM sticks) | input passthrough |
| 2 | reduce_mean | Phase 1: reduce SUM with scaler 1/W to get mean | Row-wise mean (broadcast back to full shape) |
| 3 | subtract_mean | Phase 2: subtract mean from input | x - mean(x) |
| 4 | full_layer_norm | Phases 3-5: variance, rsqrt, normalize, apply gamma+beta | Full layer_norm output |

### Stage 1: data_pipeline
- **Scope**: reader kernel (read RM sticks), compute kernel (tilize c_0 -> c_1, untilize c_1 -> c_16), writer kernel (write RM sticks)
- **Reference**: `return input_tensor` (identity passthrough)
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 32, 64)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(2, 1, 64, 64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute only does tilize+untilize. No normalization phases. Gamma/beta/scaler/eps CBs unused.
- **Notes**: This validates the full RM->tile->RM data pipeline works correctly before adding compute.

### Stage 2: reduce_mean
- **Scope**: Add reduce SUM phase in compute; reader generates scaler CB with 1/W
- **Reference**: `return input_tensor.mean(dim=-1, keepdim=True).expand_as(input_tensor)` -- outputs mean broadcast to full shape
- **Delta from previous**: Compute adds: tilize -> reduce<SUM, REDUCE_ROW> with scaler 1/W -> broadcast mean back via sub(x, mean) then add(mean_broadcast, mean_broadcast)... Actually, returning just the mean broadcast is hard. Better reference: `mean = input_tensor.mean(dim=-1, keepdim=True); return mean.expand_as(input_tensor)`
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 32, 64)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(2, 1, 64, 64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **CB bypass**: After tilize, compute reduces to mean, then broadcasts mean to all Wt output tiles (using binary_op add with zero or copy). Untilize the mean-broadcast tiles.
- **Notes**: This validates reduction + scaler are working. The compute kernel will tilize, reduce SUM (with 1/W scaler) to get mean, then use `sub` to compute (x - mean), then add mean back: output = x - (x - mean) = mean. Actually simpler: broadcast the mean tile directly to Wt output tiles.

### Stage 3: subtract_mean
- **Scope**: Compute subtracts mean from tilized input
- **Reference**: `mean = input_tensor.mean(dim=-1, keepdim=True); return input_tensor - mean`
- **Delta from previous**: Instead of broadcasting mean to output, compute does sub(tilized, mean) with COL broadcast -> untilize result
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 32, 64)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(2, 1, 64, 64)`
- **Tolerances**: rtol=0.02, atol=0.1
- **Notes**: Validates subtraction with COL broadcast works correctly.

### Stage 4: full_layer_norm
- **Scope**: All remaining phases: square, variance reduce, add eps, rsqrt, normalize, gamma, beta. Reader also loads gamma/beta.
- **Reference**: `return torch.nn.functional.layer_norm(input_tensor, [input_tensor.shape[-1]], weight=gamma.squeeze(), bias=beta.squeeze(), eps=1e-5)`
- **Delta from previous**: Adds phases 3-7 in compute. Reader adds gamma/beta reading.
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 32, 64)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(2, 1, 64, 64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel

Reads RM sticks for input tensor using TensorAccessor pattern (same as tilize reader). Per tile-row: reserves Wt pages in cb_in_rm, reads 32 sticks via noc_async_read, pushes Wt pages.

One-time setup:
- Generate reduce scaler tile into cb_scaler (c_2) using `dataflow_kernel_lib::prepare_reduce_scaler<c_2>(1.0f / W)`
- Generate epsilon tile into cb_eps (c_3) using `dataflow_kernel_lib::prepare_reduce_scaler<c_3>(epsilon)`
- Read gamma/beta: 32 repeated copies of the gamma stick into cb_gamma_rm (c_6), tilize by compute into cb_gamma (c_4). Same for beta.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out)` -- three-arg form since srcA and srcB differ.

**One-time**: Tilize gamma and beta from RM CBs (c_6, c_7) into tile CBs (c_4, c_5).

**Per tile-row loop** (num_rows_per_core iterations):

#### Phase 1: Tilize input
```cpp
compute_kernel_lib::tilize<c_0, c_1>(Wt, 1);
```
Tilizes 32 RM sticks (Wt tile-pages in c_0) into Wt tiles in c_1.

#### Phase 2: Reduce SUM for mean
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
    c_1, c_2, c_24,
    ReduceInputBlockShape::row(Wt));
```
- Input: c_1 (Wt tilized tiles, persistent via WaitUpfrontNoPop)
- Scaler: c_2 (1/W, makes SUM into MEAN)
- Output: c_24 (1 tile, mean column vector)
- Tiles in c_1 remain for Phase 3.

#### Phase 3: Subtract mean
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,   // c_1 tiles already waited
    BinaryInputPolicy::WaitAndPopPerTile,  // c_24 mean
    BinaryOutputPolicy::Bulk>(
    c_1, c_24, c_25,
    BinaryInputBlockShape::row(Wt));
```
- After: pop c_1 (Wt), pop c_24 (1). Push c_25 (Wt centered tiles).
- **Note**: c_1 tiles persist from Phase 2 (WaitUpfrontNoPop). sub uses NoWaitNoPop for c_1. c_24 mean is 1 tile broadcast as COL.

**Wait** -- sub with NoWaitNoPop on A and WaitAndPopPerTile on B (COL broadcast): The COL broadcast pops B once per row (Ht=1 here, so pops 1 tile). A is NoWaitNoPop so tiles stay. After sub, we manually pop c_1(Wt). c_25 gets Wt tiles via Bulk output.

**Revised Phase 3**: Use `WaitUpfrontPopAtEnd` for A (c_1) so it auto-pops after, and `WaitUpfrontNoPop` for B (c_24 mean, 1 tile). After sub completes, manually pop c_24(1).

```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::WaitUpfrontPopAtEnd,   // c_1: wait all Wt, pop all at end
    BinaryInputPolicy::WaitUpfrontNoPop,       // c_24: 1 tile persistent (popped manually later... or use WaitAndPopPerTile)
    BinaryOutputPolicy::Bulk>(
    c_1, c_24, c_25,
    BinaryInputBlockShape::row(Wt));
```

Actually, let's simplify. Since c_1 already has tiles waited (from reduce WaitUpfrontNoPop), and c_24 was just pushed by reduce:
- A (c_1): Use `NoWaitPopAtEnd` -- tiles already in CB, pop all Wt at end
- B (c_24): Use `WaitAndPopPerTile` -- for COL broadcast with Ht=1, waits 1 tile, pops 1 tile after the row

```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy::Bulk>(
    c_1, c_24, c_25,
    BinaryInputBlockShape::row(Wt));
```
This pops c_1(Wt) at end and c_24(1) after row. Output c_25 gets Wt tiles.

#### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,
    BinaryOutputPolicy::Bulk>(
    c_25, c_27, BinaryInputBlockShape::row(Wt));
```
- Input: c_25 (Wt centered tiles, persistent for Phase 6)
- Output: c_27 (Wt squared tiles) -- reuse c_27 temporarily
- c_25 tiles persist (WaitUpfrontNoPop) for Phase 6.

#### Phase 5: Reduce SUM for variance + add epsilon + rsqrt
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::WaitAndPopPerTile>(
    c_27, c_2, c_26,
    ReduceInputBlockShape::row(Wt),
    ReduceInputMemoryLayout::contiguous(),
    NoAccumulation{},
    [](uint32_t dst_idx) {
        // Add epsilon
        // ... need cb_eps add here
    });
```

The reduce helper's post_reduce_op gets called after the reduction with the result in DST. But we need to add epsilon (from c_3) and then rsqrt. The post_reduce_op only has access to DST, not CBs. So we need to split this:

1. Reduce SUM squared values -> c_26 (variance)
2. Binary add: c_26 + c_3 (epsilon) -> c_26 (var + eps) -- but we need separate output CB...
3. Then manually do rsqrt

**Revised Phase 5 approach**: Do the reduce, pack to c_26. Then add epsilon via binary_op. Then rsqrt via post_reduce_op or manually.

Actually, let's use the post_reduce_op for rsqrt and handle epsilon differently. We can fold epsilon into the scaler or add it manually in DST:

```cpp
// Step 5a: Reduce SUM of squares to get variance
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::WaitAndPopPerTile>(
    c_27, c_2, c_26,
    ReduceInputBlockShape::row(Wt));
// c_27 is popped (Wt tiles consumed)

// Step 5b: Add epsilon to variance: var + eps
// Use binary add: c_26 (1 tile) + c_3 (eps tile, 1 tile) -> c_26 needs separate output
// Actually output to a new temp CB or reuse c_27 (now free)
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(  // eps persists
    c_26, c_3, c_27,  // output to c_27 (free now), 1 tile
    BinaryInputBlockShape::single(),
    {},  // default layout
    {},  // no accum
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
// c_26 popped (1 tile). c_27 now has 1 tile = rsqrt(var + eps).
// But wait: c_27 was used for squared output. After reduce popped it, c_27 is free.
// Rename: use c_26 for final inv_std result.
```

This is getting complex. Let me restructure the intermediate CB usage to be cleaner.

**Revised CB usage for compute phases**:
- c_1: tilized input (Wt tiles, persists phases 2-3)
- c_24: mean (1 tile, produced in phase 2, consumed in phase 3)
- c_25: centered = x - mean (Wt tiles, produced phase 3, persists phases 4, 6)
- c_27: squared = centered^2 (Wt tiles, produced phase 4, consumed phase 5a)
- c_26: variance after reduce (1 tile, produced phase 5a, consumed phase 5b)
- c_28: inv_std = rsqrt(var + eps) (1 tile, produced phase 5b, consumed phase 6)
- c_27: (reused) normed = centered * inv_std (Wt tiles, produced phase 6, consumed phase 7)
- c_28: (reused) gamma * normed (Wt tiles, produced phase 7, consumed phase 8)
- After phase 8: add beta, untilize to c_16

Wait, we're running low on intermediate CBs. Let me restructure:

After phase 5a (reduce), c_27 is free (Wt pages consumed). We output variance to c_26 (1 tile). Phase 5b adds eps + rsqrt -> output to c_28 (1 tile). Phase 6 mul centered * inv_std -> output to c_27 (Wt tiles, reused). Phase 7 mul gamma -> output to c_28 (Wt, reused since c_28's 1-tile inv_std was consumed). Phase 8 add beta -> some output CB, then untilize to c_16.

The issue is c_28 holds 1 tile (inv_std) in phase 6, but needs Wt tiles in phase 7. CB capacity is set at init time. Solution: size c_28 to Wt tiles always (the 1-tile usage just uses less of the capacity).

**Final CB intermediate plan (simplified)**:
- c_24: mean (1 tile capacity)
- c_25: centered, x-mean (Wt tiles capacity)
- c_26: variance/inv_std scratch (1 tile capacity)
- c_27: general Wt-tile scratch (squared, then normed)
- c_28: general Wt-tile scratch (gamma*normed, then output before untilize)

Let me finalize the phase sequence with exact CB flows:

| Phase | Input CBs | Output CB | Operation | Notes |
|-------|-----------|-----------|-----------|-------|
| 1: Tilize | c_0 (Wt RM) | c_1 (Wt tile) | tilize | c_0 freed |
| 2: Mean | c_1 (Wt, persist) | c_24 (1 tile) | reduce<SUM,ROW> + scaler 1/W | c_1 persists |
| 3: Center | c_1 (Wt), c_24 (1) | c_25 (Wt) | sub<COL> | c_1 freed, c_24 freed |
| 4: Square | c_25 (Wt, persist) | c_27 (Wt) | square | c_25 persists |
| 5: Variance | c_27 (Wt) | c_26 (1) | reduce<SUM,ROW> + scaler 1/W | c_27 freed |
| 5b: Eps+Rsqrt | c_26 (1), c_3 (1) | c_26 (1) | add + rsqrt | c_3 persists, c_26 in-place |

Phase 5b is tricky: can't output back to c_26 while reading from it. Need a separate output. Use c_24 (free after phase 3):

| 5b: Eps+Rsqrt | c_26 (1), c_3 (1) | c_24 (1) | add(eps) + rsqrt post-op | c_26 freed |
| 6: Normalize | c_25 (Wt), c_24 (1) | c_27 (Wt) | mul<COL> | c_25 freed, c_24 freed |
| 7: Gamma | c_27 (Wt), c_4 (Wt) | c_28 (Wt) | mul<NONE> | c_27 freed, c_4 persists |
| 8: Beta | c_28 (Wt), c_5 (Wt) | c_27 (Wt) | add<NONE> | c_28 freed, c_5 persists |
| 9: Untilize | c_27 (Wt) | c_16 (Wt) | untilize | c_27 freed |

This works. c_24 is reused for inv_std. All CBs have enough capacity.

#### Phase 6: Normalize (multiply centered by inv_std)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,     // c_25 already waited, pop at end
    BinaryInputPolicy::WaitAndPopPerTile,  // c_24 inv_std, 1 tile
    BinaryOutputPolicy::Bulk>(
    c_25, c_24, c_27,
    BinaryInputBlockShape::row(Wt));
```

#### Phase 7: Apply gamma (element-wise multiply)
```cpp
compute_kernel_lib::mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,  // c_27 normed
    BinaryInputPolicy::NoWaitNoPop,        // c_4 gamma (persistent program-lifetime)
    BinaryOutputPolicy::PerTile>(
    c_27, c_4, c_28,
    BinaryInputBlockShape::row(Wt));
```

#### Phase 8: Apply beta (element-wise add)
```cpp
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,  // c_28
    BinaryInputPolicy::NoWaitNoPop,        // c_5 beta (persistent)
    BinaryOutputPolicy::Bulk>(
    c_28, c_5, c_27,
    BinaryInputBlockShape::row(Wt));
```

#### Phase 9: Untilize
```cpp
compute_kernel_lib::untilize<Wt, c_27, c_16>(1);
```

### Writer Kernel

Writes RM sticks from cb_out (c_16) to DRAM. Per tile-row: waits for Wt pages in c_16, extracts 32 sticks (each of width W elements), writes via TensorAccessor, pops Wt pages.

Pattern matches the untilize writer: for each block, wait for Wt tiles in c_16, iterate 32 rows extracting stick-sized chunks, write each stick via noc_async_write using TensorAccessor.

### Critical Notes

1. **Gamma/beta tilization**: Must happen ONCE at the start of compute, before the per-row loop. Reader fills c_6/c_7 with 32 copies of the gamma/beta stick. Compute tilizes c_6->c_4 and c_7->c_5. These persist for the entire program (never popped from c_4/c_5 after initial tilize).

2. **Scaler tile format**: The reduce scaler must be in the specific face-interleaved format required by the reduce hardware. Use `dataflow_kernel_lib::prepare_reduce_scaler<cb_id>(1.0f / W)` in the reader kernel.

3. **Epsilon addition**: Cannot use the reduce scaler for epsilon. Epsilon is added separately via binary_op::add after the variance reduction. The epsilon tile is also generated by `prepare_reduce_scaler` but for a different purpose -- it's a general-purpose tile fill.

4. **CB c_25 persistence**: Centered values (x - mean) must persist through phases 4 and 6. Phase 4 (square) uses WaitUpfrontNoPop. Phase 6 (normalize) uses NoWaitPopAtEnd.

5. **Untilize block_width_tiles**: The untilize helper takes `block_width_tiles` as a compile-time template parameter. Since Wt may vary per invocation, this needs to be a compile-time arg. The program factory should set Wt as a compile define.

6. **hw_startup with 3 args**: Since we use reduce (needs scaler as srcB) and binary ops (needs different srcB), we need `compute_kernel_hw_startup(c_1, c_2, c_16)` or similar. The binary_op helpers handle re-init internally.

### Implementation Checklist
- [ ] Reader: TensorAccessor for input/gamma/beta sticks, prepare_reduce_scaler for c_2 and c_3
- [ ] Compute: 9 phases using helpers: tilize, reduce<SUM,ROW>, sub<COL>, square, reduce<SUM,ROW>, add+rsqrt, mul<COL>, mul<NONE>, add<NONE>, untilize
- [ ] Writer: TensorAccessor for output sticks, untilize stick extraction pattern
- [ ] CB push/pop balance verified per phase
