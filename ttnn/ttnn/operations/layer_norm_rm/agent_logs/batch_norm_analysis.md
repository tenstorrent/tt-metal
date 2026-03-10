# Batch Norm Implementation Analysis (Compute-Core Reference)

## Overview

Batch normalization computes `output = ((input - mean) / sqrt(var + eps)) * gamma + beta` for each element of the input tensor, where mean and variance are pre-computed per-channel statistics provided as separate input tensors. The optional gamma (weight) and beta (bias) are per-channel affine parameters.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Key architectural note for layer_norm_rm reference**: Batch norm receives pre-computed mean and variance as inputs (no reduction needed in compute kernel). A layer_norm_rm operation will need to compute mean and variance via row-wise reduction within the compute kernel itself, which is a significant structural difference. This analysis focuses on the normalization, rsqrt, and affine transform phases that are directly reusable.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Outer loop over channel groups (complete_iterations), inner loop over tiles within a channel group (tile_start to freq) |

The compute kernel processes one tile at a time. A "channel group" consists of `HtWt = Ht * Wt` tiles that share the same per-channel mean/variance/weight/bias. The `freq` runtime arg equals `cHtWt` (tiles per channel slice), and `counter` equals `start_tile_id % cHtWt` (offset into the first channel group for the core).

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | Batch Mean | Batch Var | Weight (gamma) | Bias (beta) |
|----------|-----------|------------|-----------|----------------|-------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16/FLOAT32 | BFLOAT16/FLOAT32 | BFLOAT16/FLOAT32 | BFLOAT16/FLOAT32 | BFLOAT16/FLOAT32 |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Configurable (defaults to input dtype) |

### Layout Transformations

The writer kernel uses `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast a single scalar value across an entire tile for mean, variance, weight, and bias. These are per-channel scalars stored as the first element of a tile in DRAM; the writer reads the tile and fills all 1024 elements with that first element, creating a uniform broadcast tile. This is a **data-movement-level broadcast** done before compute sees the data.

## Data Flow Pattern

### High-Level Flow

```
DRAM(input) --[reader]--> CB_input --[compute]--> CB_output --[writer]--> DRAM(output)
DRAM(eps)   --[reader]--> CB_eps   --[compute]-->  (consumed once, persists entire program)
DRAM(mean)  --[writer]--> CB_mean  --[compute]-->  (consumed per channel group)
DRAM(var)   --[writer]--> CB_var   --[compute]-->  (consumed per channel group)
DRAM(weight)--[writer]--> CB_weight--[compute]-->  (consumed per channel group)
DRAM(bias)  --[writer]--> CB_bias  --[compute]-->  (consumed per channel group)
                                      CB_den   (intermediate: rsqrt result, per channel group)
                                      CB_tmp_1 (intermediate: scratch for affine pipeline)
```

### Detailed Compute Steps (per channel group)

**Phase 1: Compute rsqrt (once per channel group)**
1. Wait for `cb_batch_var` (1 tile from writer)
2. Wait for `cb_eps` (already present from reader, persists across program)
3. Compute `var + eps` via `add_tiles`
4. Compute `rsqrt(var + eps)` via `rsqrt_tile`
5. Pack result to `cb_den`
6. Pop `cb_batch_var`

**Phase 2: Normalize each tile in the group (repeated `freq` times)**
7. Wait for `cb_input` (1 tile from reader)
8. Wait for `cb_batch_mean` (already present, persists across channel group)
9. Wait for `cb_den` (already present, persists across channel group)
10. Compute `input - mean` via `sub_tiles`
11. Compute `(input - mean) * rsqrt` via `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` (FPU kernel) or `mul_binary_tile` (SFPU kernel)
12. Pack result to `cb_affine_or_out` (either `cb_tmp_1` if affine needed, or `cb_output` if not)
13. Pop `cb_input`

**Phase 3: Apply gamma (conditional, per tile)**
14. If weight: wait `cb_affine_or_out`, compute `result * weight` via `mul_tiles`, pack to `cb_scaled_output`

**Phase 4: Apply beta (conditional, per tile)**
15. If bias: wait `cb_tmp_1`, compute `result + bias` via `add_tiles`, pack to `cb_output`

**Phase 5: End of channel group**
16. Pop `cb_batch_mean`, `cb_den`, and optionally `cb_weight`, `cb_bias`

### CB Routing for Affine Transform

The compute kernel dynamically routes output through different CBs depending on which affine parameters are present:

```
cb_affine_or_out = (weight || bias) ? cb_tmp_1 : cb_output
cb_scaled_output = (bias) ? cb_tmp_1 : cb_output
```

This means:
- **No gamma, no beta**: normalized result goes directly to `cb_output`
- **Gamma only**: normalized goes to `cb_tmp_1`, after multiply goes to `cb_output`
- **Beta only**: normalized goes to `cb_tmp_1`, after add goes to `cb_output`
- **Gamma and beta**: normalized goes to `cb_tmp_1`, after multiply stays in `cb_tmp_1`, after add goes to `cb_output`

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tiles (x) | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | cb_batch_mean | Broadcast mean tile | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel group) |
| c_2 | cb_output | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | cb_batch_var | Broadcast variance tile | 2 tiles | 1 tile | Double | Writer | Compute | Block (per channel group, consumed in rsqrt) |
| c_4 | cb_eps | Epsilon scalar tile | 2 tiles | 1 tile | Double | Reader | Compute | Program (filled once, read entire program) |
| c_5 | cb_weight | Broadcast gamma tile | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel group) |
| c_6 | cb_bias | Broadcast beta tile | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel group) |
| c_7 | cb_den | rsqrt(var+eps) intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Row (per channel group, self-produced/consumed) |
| c_8 | cb_tmp_1 | Affine transform scratch | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile, self-produced/consumed) |

### Key CB Lifetime Notes for Layer Norm Reference

1. **cb_eps (c_4)**: Program-lifetime CB. Filled once by reader at startup, persisted via single `cb_wait_front` before main loop and single `cb_pop_front` after all loops complete. The compute kernel never pops it during processing. This is the canonical pattern for constant scalars.

2. **cb_batch_mean (c_1)** and **cb_den (c_7)**: Channel-group-lifetime CBs. They are `cb_wait_front`-ed at the start of each channel group iteration and `cb_pop_front`-ed only when all tiles in the group are processed. This is the broadcast pattern: one scalar tile is reused across multiple input tiles.

3. **cb_input (c_0)** and **cb_output (c_2)**: Per-tile lifetime. Each tile is waited/popped individually.

4. **cb_tmp_1 (c_8)**: Self-produced and self-consumed by compute. Used as an intermediate staging buffer between normalization and affine transforms.

## Pipeline Pattern Summary

All CBs have capacity=2 tiles and block_size=1 tile, making them all **double-buffered**. This allows overlap between producer and consumer: while compute processes one tile, the reader/writer can prefetch/drain the next.

## Index Calculations

### Tile Frequency and Counter (Compute Kernel)

The compute kernel uses `tile_freq` and `tile_start` to determine channel group boundaries:

```cpp
// In program factory:
auto counter = start_tile_id % cHtWt;  // offset into first channel group
auto freq = cHtWt;                       // tiles per channel = Ht * Wt

// In compute kernel:
uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
```

- `complete_iterations`: number of full channel groups this core processes
- `remaining_iterations`: leftover tiles in a partial channel group at the end
- `tile_start`: starts at `counter`, then resets to 0 for subsequent groups

This scheme handles the case where a core's tile range does not align with channel boundaries. The first channel group may be partial (starting at `tile_start`), middle groups are complete, and the last may be partial.

### Reader/Writer Index Calculations

The reader iterates N/C/HW nested loops with stride-based offsets:
```cpp
tile_offset = start_n * n_stride + start_c * c_stride + start_t;
next_channel_shift = c_stride - HtWt;
next_batch_shift = n_stride - c_stride * C;
```

The writer uses the same N/C structure but reads per-channel broadcast tiles at channel boundaries and writes output tiles within each channel.

## Memory Access Patterns

### Read Pattern (Reader Kernel)
- **Input tensor**: Sequential tile reads within each HxW spatial block, then stride to next channel, then stride to next batch. Pattern is `[N][C][HtWt]` nested.
- **Epsilon**: Single fill operation at start, never re-read.

### Read Pattern (Writer Kernel - reading mean/var/weight/bias)
- Per-channel tiles read once per channel group, then broadcast-filled in L1.
- Sequential within each channel group: read broadcast tile, then iterate HtWt output tiles.

### Write Pattern (Writer Kernel)
- Sequential tile writes indexed by `start_tile_id + num_tiles_written` using TensorAccessor.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (but linearized as row-major) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores_x * num_cores_y` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` |
| **Load balancing** | Two-group split via `split_work_to_cores` (core_group_1 gets `ceil`, core_group_2 gets `floor`) |

The `split_work_to_cores` utility divides `num_output_tiles` across available cores. Cores in `core_group_1` get one extra tile compared to `core_group_2`. Cores outside both groups get zero-arg runtime args and exit immediately (`if (num_tiles == 0) return;`).

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if gamma weight tensor is provided, else 0 |
| 1 | bias_has_value | uint32_t | 1 if beta bias tensor is provided, else 0 |
| 2 | cb_input | uint32_t | CB index for input (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for broadcast mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon (c_4) |
| 7 | cb_den | uint32_t | CB index for rsqrt intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for gamma weight (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for scratch intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB index for beta bias (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles this core must process |
| 1 | tile_freq | uint32_t | Tiles per channel group (cHt * cWt) = broadcast reuse frequency |
| 2 | tile_start | uint32_t | Offset into first channel group (start_tile_id % cHtWt) |

### Compile-Time Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src | uint32_t | CB index for input (c_0) |
| 1 | cb_id_eps | uint32_t | CB index for epsilon (c_4) |
| 2+ | TensorAccessorArgs | auto | Accessor parameters for input tensor |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon as packed bf16 pair or float32 bits |
| 1 | src_addr | uint32_t | Input tensor DRAM address |
| 2 | start_tile_id | uint32_t | Starting output tile for this core |
| 3 | num_tiles | uint32_t | Total tiles for this core |
| 4 | HtWt | uint32_t | cHt * cWt (tiles per channel) |
| 5 | n_stride | uint32_t | Tile stride between batches |
| 6 | c_stride | uint32_t | Tile stride between channels |
| 7 | N | uint32_t | Number of batches |
| 8 | C | uint32_t | Number of channels |
| 9 | Ht | uint32_t | Height in tiles |
| 10 | Wt | uint32_t | Width in tiles |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Mean tensor DRAM address |
| 1 | batch_var_addr | uint32_t | Variance tensor DRAM address |
| 2 | weight_addr | uint32_t | Weight tensor DRAM address (0 if absent) |
| 3 | bias_addr | uint32_t | Bias tensor DRAM address (0 if absent) |
| 4 | dst_addr | uint32_t | Output tensor DRAM address |
| 5 | start_tile_id | uint32_t | Starting output tile for this core |
| 6 | num_tiles | uint32_t | Total tiles for this core |
| 7-13 | HtWt, n_stride, c_stride, N, C, Ht, Wt | uint32_t | Shape/stride info (same semantics as reader) |

## Kernel Implementations

### Kernel Specification Table

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM(input) | CB_input(c_0), CB_eps(c_4) | Read input tiles sequentially, fill epsilon tile once |
| compute (batch_norm_kernel / batch_norm_sfpu_kernel) | RISCV_2 | N/A | CB_input, CB_mean, CB_var, CB_eps, CB_weight, CB_bias | CB_output(c_2), CB_den(c_7), CB_tmp(c_8) | add, rsqrt, sub, mul, pack (normalization pipeline) |
| writer_batch_norm | RISCV_1 | NOC1 | DRAM(mean,var,weight,bias), CB_output | DRAM(output), CB_mean(c_1), CB_var(c_3), CB_weight(c_5), CB_bias(c_6) | Read broadcast tiles, fill_tile, write output |

### Compute Kernel: FPU Path (`batch_norm_kernel.cpp`)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`

**Includes**: `api/compute/eltwise_binary.h`, `ttnn/kernel/compute/moreh_common.hpp`

**Initialization**:
```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```
This configures the FPU unpacker, math, and packer units for binary operations. Called once at kernel start.

**Key Compute Sequence (FPU path, per channel group)**:

1. **rsqrt computation** (var + eps -> 1/sqrt):
   ```cpp
   tile_regs_acquire();
   add_tiles_init_with_dt(cb_batch_var, cb_eps);
   add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);  // dst0 = var + eps
   rsqrt_tile_init();
   rsqrt_tile(dst0);                              // dst0 = 1/sqrt(var + eps)
   tile_regs_commit();
   tile_regs_wait();
   pack_tile_with_dt(dst0, cb_den);               // pack to cb_den
   tile_regs_release();
   ```

2. **Subtract mean and multiply by rsqrt** (using dest reuse for zero-copy chaining):
   ```cpp
   tile_regs_acquire();
   sub_tiles_init(cb_other, cb_bcast);
   sub_tiles(cb_other, cb_bcast, 0, 0, 0);        // dst0 = input - mean
   binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
   binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
   // dst0 = (input - mean) * rsqrt  -- no intermediate CB write!
   tile_regs_commit();
   tile_regs_wait();
   pack_tile_with_dt(0, cb_affine_or_out);
   tile_regs_release();
   ```

   **Critical pattern**: `binary_dest_reuse_tiles` avoids writing the `(input - mean)` intermediate to a CB. The result of `sub_tiles` remains in DST register 0, which is then moved to SRCA for the multiply with `cb_den`. This is a key optimization for chaining operations.

3. **Gamma multiply** (conditional):
   ```cpp
   tile_regs_acquire();
   mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
   mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
   tile_regs_commit();
   tile_regs_wait();
   pack_tile_with_dt(dst0, cb_scaled_output);
   tile_regs_release();
   ```

4. **Beta add** (conditional):
   ```cpp
   tile_regs_acquire();
   add_tiles_init_with_dt(cb_tmp_1, cb_bias);
   add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
   tile_regs_commit();
   tile_regs_wait();
   pack_tile_with_dt(dst0, cb_output_0);
   tile_regs_release();
   ```

### Compute Kernel: SFPU Path (`batch_norm_sfpu_kernel.cpp`)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`

**Includes**: `api/compute/eltwise_binary_sfpu.h`, `api/compute/eltwise_unary/sfpu_split_includes.h`, `api/compute/eltwise_unary/eltwise_unary.h`, `api/compute/eltwise_unary/rsqrt.h`

**Initialization**:
```cpp
unary_op_init_common(cb_other, cb_output_0);
```

**Key difference from FPU path**: SFPU operations use explicit `copy_tile` to load data into DST registers before computing, and use `*_binary_tile` functions that operate on DST register indices rather than CB indices.

**SFPU rsqrt computation**:
```cpp
tile_regs_acquire();
tile_regs_wait();
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
copy_tile(cb_batch_var, 0, 0);       // dst[0] = var
add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
copy_tile(cb_eps, 0, 1);             // dst[1] = eps
add_binary_tile(0, 1, 0);            // dst[0] = var + eps
rsqrt_tile_init();
rsqrt_tile(0);                       // dst[0] = 1/sqrt(var + eps)
pack_tile(0, cb_den);
tile_regs_commit();
tile_regs_release();
```

**SFPU sub + mul (per tile)**:
```cpp
tile_regs_acquire();
tile_regs_wait();
copy_tile_to_dst_init_short_with_dt(cb_bcast, cb_other);
copy_tile(cb_other, 0, 0);           // dst[0] = input
sub_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_other, cb_bcast);
copy_tile(cb_bcast, 0, 1);           // dst[1] = mean
sub_binary_tile(0, 1, 0);            // dst[0] = input - mean
mul_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_bcast, cb_den);
copy_tile(cb_den, 0, 1);             // dst[1] = rsqrt
mul_binary_tile(0, 1, 0);            // dst[0] = (input - mean) * rsqrt
pack_tile(0, cb_affine_or_out);
tile_regs_commit();
tile_regs_release();
```

**Key SFPU pattern notes**:
- Uses `copy_tile_to_dst_init_short_with_dt` to reconfigure unpacker between different CB data formats
- Operations work on DST register indices (0, 1) rather than CB indices
- `tile_regs_acquire()` + `tile_regs_wait()` are called back-to-back (SFPU path acquires and immediately waits)
- Multiple operations chain in DST without intermediate CB writes (similar optimization to FPU dest reuse)

### Kernel Selection (FPU vs SFPU)

Selected at program factory time based on `fp32_dest_acc_en`:
```cpp
auto compute_kernel_id = tt_metal::CreateKernel(
    program,
    fmt::format("...batch_norm_{}.cpp",
        fp32_dest_acc_en ? "sfpu_kernel" : "kernel"),
    all_device_cores,
    tt_metal::ComputeConfig{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .compile_args = compute_kernel_args});
```

When `fp32_dest_acc_en` is true:
- SFPU kernel is used
- All compute CBs get `UnpackToDestMode::UnpackToDestFp32`
- Maintains full FP32 precision in DST accumulator

## Implementation Notes

### Multi-Pass Data Reuse Pattern (Critical for Layer Norm Reference)

The key reuse pattern in batch_norm's compute kernel is the **channel-group broadcast**:

1. **cb_eps**: Filled once, persists entire program. Pattern:
   ```cpp
   cb_wait_front(cb_eps, onetile);  // Before main loop
   // ... all processing ...
   cb_pop_front(cb_eps, onetile);   // After main loop
   ```

2. **cb_batch_mean, cb_den, cb_weight, cb_bias**: Filled once per channel group, reused for all HtWt tiles in the group. Pattern:
   ```cpp
   cb_wait_front(cb_bcast, onetile);  // Start of channel group
   cb_wait_front(cb_den, onetile);
   for (j = tile_start; j < freq; ++j) {
       // Process each input tile using same bcast/den/weight/bias
   }
   cb_pop_front(cb_bcast, onetile);   // End of channel group
   cb_pop_front(cb_den, onetile);
   ```

**For layer_norm_rm**: The analogous pattern would be:
- Compute mean via reduce (accumulate across tiles in a row), store in a CB
- Compute centered values, accumulate variance via reduce
- Compute rsqrt of variance (like `cb_den` here)
- Then broadcast rsqrt across all tiles in the row
- The mean and rsqrt CBs would have **row lifetime** (persisting across all tiles in one normalization row)

### Scalar/Constant CB Setup Pattern

Epsilon is set up as a full tile filled with the scalar value:
```cpp
// Reader kernel:
cb_reserve_back(cb_id_eps, onetile);
FILL_WITH_VALUE_FLOAT(cb_id_eps, scalar.f);   // or FILL_WITH_VALUE for bf16
cb_push_back(cb_id_eps, onetile);
```

The `FILL_WITH_VALUE_FLOAT` expands to `fill_with_val<1024, float>`, writing 1024 float elements (one full 32x32 tile). For bf16, `fill_with_val_bfloat16` writes 512 packed uint32 values (1024 bf16 elements).

**For layer_norm_rm**: The same pattern can be used for epsilon. An additional scalar CB may be needed for `1/W` (the reduction scaling factor), which can be set up the same way.

### Binary Op Broadcast Pattern (FPU Dest Reuse)

The FPU kernel chains subtraction and multiplication without an intermediate CB write:
```cpp
sub_tiles(cb_other, cb_bcast, 0, 0, 0);  // result stays in DST[0]
binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_den, 0, 0);  // DST[0] * cb_den[0]
```

`DEST_TO_SRCA` means DST[0] is loaded into SRCA, and `cb_den[0]` is loaded into SRCB via the unpacker. The multiply result goes back to DST[0].

**For layer_norm_rm**: This pattern is directly applicable for `(x - mean) * rsqrt`. The dest reuse avoids allocating an extra CB for the centered intermediate.

### Reduce Helper Parameters (Not Used in Batch Norm, but Needed for Layer Norm)

From the `reduce.h` API (documented above), the key reduce function signatures are:

```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb);

template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst);

template <bool enforce_fp32_accumulation = false>
void reduce_uninit(uint32_t icb = 0);
```

For layer_norm_rm row-wise mean:
- `PoolType::SUM` with `ReduceDim::REDUCE_ROW` reduces a 32x32 tile to column values (result in first row)
- `icb_scaler` should contain `1.0` for sum, or `1/W` for averaging
- Multiple tiles can be accumulated into the same DST register by calling `reduce_tile` in a loop without releasing DST
- After reduction, `reduce_uninit` must be called to reset packer edge masks

### UnpackToDestMode for FP32 Precision

When `fp32_dest_acc_en` is true, all compute-relevant CBs are set to `UnpackToDestMode::UnpackToDestFp32`:
```cpp
if (fp32_dest_acc_en) {
    for (const auto cb_index : {input_cb, mean_cb, var_cb, eps_cb, den_cb, weight_cb, tmp_cb, bias_cb}) {
        unpack_to_dest_mode[cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }
}
```

This ensures FP32 accumulation in DST registers, which is important for numerical stability in normalization operations.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What are the compute kernel APIs used in tt-metal? Specifically explain: tile_regs_acquire, tile_regs_commit, tile_regs_wait, tile_regs_release, add_tiles, sub_tiles, mul_tiles, rsqrt_tile, pack_tile_with_dt, binary_op_init_common, binary_dest_reuse_tiles"
   **Reason**: Needed to understand the FPU compute pipeline and register lifecycle
   **Key Findings**: tile_regs_acquire acquires DST registers for math and zeros them; tile_regs_commit transfers ownership to packer; tile_regs_wait lets packer wait for results; tile_regs_release frees DST for next iteration. FPU ops (add_tiles, mul_tiles) operate directly on CBs; SFPU ops require explicit copy_tile into DST.

2. **Query**: "Difference between add_tiles (FPU) vs add_binary_tile (SFPU), copy_tile usage, copy_tile_to_dst_init_short_with_dt"
   **Reason**: Batch norm has two kernel variants (FPU and SFPU) using different APIs
   **Key Findings**: FPU path works directly with CB indices; SFPU path requires explicit copy_tile to load data into DST registers before computing. copy_tile_to_dst_init_short_with_dt reconfigures the unpacker for different data formats. SFPU enables easier chaining in DST.

3. **Query**: "How do reduce operations work in tt-metal compute kernels? reduce_init, reduce_tile, ReduceDim::REDUCE_ROW, PoolType::SUM"
   **Reason**: Layer_norm_rm will need row-wise reduction for mean and variance computation (not present in batch_norm)
   **Key Findings**: reduce_init configures HW for reduce op with icb, icb_scaler (scaling factors), ocb. reduce_tile accumulates into DST. REDUCE_ROW reduces across rows to column vector (result in first row). icb_scaler should contain 1.0 for SUM or 1/N for AVG. reduce_uninit must be called after to reset packer masks.

4. **Query**: "How does binary_dest_reuse_tiles work? DEST_TO_SRCA vs DEST_TO_SRCB"
   **Reason**: Batch norm uses this for chaining sub+mul without intermediate CB write
   **Key Findings**: DEST_TO_SRCA loads DST into SRCA, reads CB into SRCB. DEST_TO_SRCB does the reverse. Key optimization: avoids intermediate CB writes when chaining operations, keeping data in fast DST registers.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: Full API signatures and documentation for reduce operations needed by layer_norm_rm
   **Key Information**: `reduce_init<PoolType, ReduceDim, enforce_fp32>`, `reduce_tile` signature with icb, icb_scaler, itile, itile_scaler, idst parameters. reduce_uninit required to clear packer masks after reduce.

2. **Source**: `tt_metal/hw/inc/api/compute/bcast.h`
   **Reason**: Understanding broadcast operations for sub_tiles_bcast, mul_tiles_bcast patterns
   **Key Information**: BroadcastType::ROW broadcasts a row across all rows of a tile (B[w] applied to A[h,w]). BroadcastType::COL broadcasts column. BroadcastType::SCALAR broadcasts single value. init_bcast<op,dim> configures HW.

3. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Helper functions used by batch_norm and available for layer_norm_rm
   **Key Information**: Provides `_with_dt` wrapper variants (add_tiles_init_with_dt, mul_tiles_init_with_dt, etc.) that handle FP32_DEST_ACC_EN reconfig. Also provides composite helpers like mul_tiles_to_cb, add_tiles_to_cb, sub_tiles_bcast_cols_to_cb, mul_tiles_bcast_rows_to_cb, copy_tile_to_cb, recip_tile_to_cb.

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Understanding how scalar constants are turned into full tiles for compute
   **Key Information**: fill_with_val_bfloat16 fills 512 uint32 (1024 bf16 elements). fill_tile_with_first_element_bfloat16 reads first element and fills entire tile. fill_tile_with_first_row copies first row to all rows (useful for row-broadcast after reduce).

5. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: binary_dest_reuse_tiles signature and semantics
   **Key Information**: `binary_dest_reuse_tiles_init<EltwiseBinaryType, EltwiseBinaryReuseDestType>(icb0)` takes one CB; `binary_dest_reuse_tiles<type, reuse>(in_cb_id, in_tile_index, dst_tile_index)` performs operation with DST as one operand and CB as the other.

6. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding create_cb helper used in program factory
   **Key Information**: `create_cb(cb_index, program, core_spec, page_size, num_pages, data_format)` returns `tuple<uint32_t, CBHandle>`. Configures CircularBufferConfig with page_size * num_pages total size.
