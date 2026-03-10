# Batch Norm Implementation Analysis

## Overview

Batch normalization computes `output = (input - batch_mean) / sqrt(batch_var + eps)`, with optional affine transformation `output = output * weight + bias`. The mean and variance are provided as pre-computed per-channel tensors (not computed within this kernel), so this operation is purely the normalize-scale-shift pipeline.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Compute kernel paths** (two variants selected at kernel creation time):
- FPU path: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- SFPU path: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`

The FPU kernel is used when `fp32_dest_acc_en == false`; the SFPU kernel is used when `fp32_dest_acc_en == true`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` (total tiles in output tensor) |
| **Loop structure** | Outer loop over "channels" (groups of `HtWt` tiles sharing the same mean/var/weight/bias), inner loop over spatial tiles within each channel |

The key concept is `tile_freq = cHt * cWt`, which is the number of spatial tiles per channel. The per-channel parameters (mean, variance, weight, bias) are loaded once per `tile_freq` tiles, then reused across all spatial tiles in that channel group. The compute kernel tracks position within this frequency via `tile_start` and `complete_iterations/remaining_iterations`.

## Tensor Format and Layout

### Input Tensors

| Property | Input (`input`) | Batch Mean | Batch Var | Weight (optional) | Bias (optional) |
|----------|-----------------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [N, C, 1, 1] or [1, C, 1, 1] | same as mean | same as mean | same as mean |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 |

### Layout Transformations

The per-channel tensors (mean, var, weight, bias) are 1D in their effective semantics (one scalar per channel), but stored as tiles. The writer kernel reads each such tile and calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the single scalar value across all 1024 elements in the tile. This converts a scalar-per-channel value into a full tile suitable for element-wise tile operations in compute. No tilize/untilize is performed; all tensors are already in TILE_LAYOUT.

## Data Flow Pattern

### Phase 1: Initialization (once per program)
1. **Reader** fills epsilon CB (`c_4`) with a constant tile: every element in the tile is set to `eps`. Uses `fill_with_val` or `fill_with_val_bfloat16`. This tile persists for the entire kernel execution.

### Phase 2: Per-channel setup (once per `HtWt` tiles)
2. **Writer** reads one tile of `batch_mean` from DRAM, calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast it, pushes to `c_1`.
3. **Writer** reads one tile of `batch_var` from DRAM, calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast it, pushes to `c_3`.
4. **Writer** (if weight exists) reads one tile of `weight` from DRAM, fills with first element, pushes to `c_5`.
5. **Writer** (if bias exists) reads one tile of `bias` from DRAM, fills with first element, pushes to `c_6`.
6. **Compute** receives `batch_var` from `c_3` and `eps` from `c_4`, computes `rsqrt(batch_var + eps)`, stores result in `c_7` (den). This `den` tile persists across all spatial tiles in the channel.

### Phase 3: Per-tile computation (inner loop, `HtWt` iterations per channel)
7. **Reader** reads one input tile from DRAM, pushes to `c_0`.
8. **Compute** receives input from `c_0` and mean from `c_1`, computes `input - batch_mean`, then multiplies by `den` from `c_7`: `(input - mean) * rsqrt(var + eps)`. Result goes to `c_8` (if affine) or `c_2` (if no affine).
9. **Compute** (if weight) multiplies result by `weight` from `c_5`, output to `c_8` (if bias) or `c_2` (if no bias).
10. **Compute** (if bias) adds `bias` from `c_6` to result, output to `c_2`.
11. **Writer** reads completed tile from `c_2`, writes to DRAM output.

### Phase 4: Channel boundary
12. After all `HtWt` tiles for a channel, compute pops mean (`c_1`), den (`c_7`), weight (`c_5`), bias (`c_6`). Flow returns to Phase 2 for the next channel.

**Important naming caveat**: The "writer" kernel is responsible for *reading* the per-channel parameter tensors (mean, var, weight, bias) from DRAM and *writing* output tiles to DRAM. The "reader" kernel reads only the input tensor and fills the epsilon constant.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| `c_0` | `input_tensor_cb` | Input tiles from DRAM | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| `c_1` | `batch_mean_tensor_cb` | Broadcast mean tile | 2 tiles | 1 tile | Double | Writer | Compute | Row (persists across `HtWt` tiles per channel) |
| `c_2` | `output_tensor_cb` | Final output tile | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| `c_3` | `batch_var_tensor_cb` | Broadcast variance tile | 2 tiles | 1 tile | Double | Writer | Compute | Block (consumed once per channel to compute den) |
| `c_4` | `eps_cb` | Epsilon constant tile | 2 tiles | 1 tile | Double | Reader | Compute | Program (filled once, persists entire execution) |
| `c_5` | `weight_tensor_cb` | Broadcast weight tile | 2 tiles | 1 tile | Double | Writer | Compute | Row (persists across `HtWt` tiles per channel) |
| `c_6` | `bias_tensor_cb` | Broadcast bias tile | 2 tiles | 1 tile | Double | Writer | Compute | Row (persists across `HtWt` tiles per channel) |
| `c_7` | `den_cb` | `1/sqrt(var + eps)` intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Row (persists across `HtWt` tiles per channel) |
| `c_8` | `temp_1_cb` | Intermediate result for affine transform | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile) |

### Multi-pass Data Reuse Patterns

The following CBs persist across multiple tiles within a channel group (across the inner `HtWt` loop):

- **`c_1` (batch_mean)**: Pushed once per channel by writer, waited-on at the start of `batchnorm_bcast_tiles`, popped only at the end after all `freq` tiles are processed.
- **`c_4` (eps)**: Pushed once at program start by reader, waited-on in `kernel_main`, popped only at program end. Persists across ALL channels and ALL batches.
- **`c_5` (weight)**: Same lifetime as `c_1` -- waited-on at function entry, popped at function exit.
- **`c_6` (bias)**: Same lifetime as `c_1`.
- **`c_7` (den)**: Computed once per channel by compute from var+eps, then reused across all `HtWt` tiles. Popped at end of `batchnorm_bcast_tiles`.

CBs that are consumed per-tile (not reused):
- **`c_0` (input)**: One tile pushed per iteration by reader, one popped per iteration by compute.
- **`c_2` (output)**: One tile pushed per iteration by compute, one popped per iteration by writer.
- **`c_3` (batch_var)**: Consumed once per channel to compute `den`, not reused.
- **`c_8` (temp_1)**: Transient; produced and consumed within a single tile's affine computation.

### Scalar/Constant CB Setup

**Epsilon (`c_4`)**: The reader kernel receives the packed epsilon value as runtime arg 0. It calls `fill_with_val` (FP32 path) or `fill_with_val_bfloat16` (BF16 path) to fill an entire tile (1024 elements) with the epsilon scalar. This tile is then pushed to `c_4` and never popped until program end. The compute kernel calls `cb_wait_front(cb_eps, onetile)` once before the main loop and `cb_pop_front(cb_eps, onetile)` after all processing.

**Per-channel broadcasts (mean, var, weight, bias)**: The writer reads a tile from DRAM (which stores the per-channel scalar in the first element), then calls `FILL_TILE_WITH_FIRST_ELEMENT` to replicate that scalar across the entire tile. This is a dataflow-side operation (pure L1 memory manipulation, no compute involved). The define `FILL_TILE_WITH_FIRST_ELEMENT` resolves to either `fill_tile_with_first_element_bfloat16` or `fill_tile_with_first_element<float>` depending on data type.

## Pipeline Pattern Summary

All CBs have capacity = 2 tiles and block size = 1 tile, yielding double-buffered semantics. This means:
- Reader can push the next input tile while compute is still processing the current one.
- Compute can push the next output tile while writer is still writing the previous one.
- The double buffering on `c_1`, `c_5`, `c_6`, `c_7` is available capacity but not heavily leveraged for overlap since these are held for multi-tile reuse.

## Index Calculations

### Output tile to (N, C, t) mapping

The program factory computes `cHtWt = cHt * cWt` (tiles per channel spatial region). Given a linear `start_tile_id`:

```
tiles_per_batch = HtWt * C
start_n = start_tile_id / tiles_per_batch
start_remaining = start_tile_id % tiles_per_batch
start_c = start_remaining / HtWt
start_t = start_remaining % HtWt
```

### Compute kernel frequency tracking

The compute kernel receives:
- `num_tiles`: total tiles this core processes
- `tile_freq`: equals `cHtWt` (tiles per channel)
- `tile_start`: equals `start_tile_id % cHtWt` (offset within current channel)

It then computes:
```
complete_iterations = (num_tiles + tile_start) / tile_freq
remaining_iterations = (num_tiles + tile_start) % tile_freq
```

Each `complete_iteration` calls `batchnorm_bcast_tiles` with the full `freq` range (starting from `tile_start` in the first iteration, 0 afterward). The remainder handles the partial last channel.

### Reader tile offset calculation

The reader uses stride-based addressing:
```
tile_offset = start_n * n_stride + start_c * c_stride + start_t
next_channel_shift = c_stride - HtWt
next_batch_shift = n_stride - c_stride * C
```

Where `n_stride = aHt * aWt * aC * (aN > 1)` and `c_stride = aHt * aWt * (aC > 1)`. The `(aN > 1)` / `(aC > 1)` guards handle broadcasting when input has singleton batch/channel dimensions.

### Writer tile offset calculation

The writer uses the `b`-shape strides (from batch_mean/batch_var shape) for reading per-channel parameters:
```
tile_offset = start_n * n_stride + start_c * c_stride
```

Note: no `start_t` offset for per-channel parameters since they have no spatial dimension.

## Memory Access Patterns

### Read Pattern
- **Input (reader)**: Sequential within each channel (t=0..HtWt-1), then skips by `next_channel_shift` to next channel, then skips by `next_batch_shift` to next batch. Effectively NCHW-ordered tile reads with stride adjustments for padded dimensions.
- **Per-channel params (writer)**: One tile read per channel, sequential across channels within a batch. Each read is followed by `FILL_TILE_WITH_FIRST_ELEMENT` (scalar broadcast) in L1.

### Write Pattern
- **Output (writer)**: Sequential tile writes using `start_tile_id + num_tiles_written` as the linear tile index into the output tensor accessor. Monotonically increasing.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (but linearized as 1D row-major) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | Up to full device grid (e.g., 8x8 = 64) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` output tiles |
| **Load balancing** | Two-group split: group_1 gets `ceil(total/cores)` tiles, group_2 gets `floor(total/cores)` tiles |

Uses `tt::tt_metal::split_work_to_cores` with `row_major = true`. Cores not assigned any tiles receive zero-initialized runtime args and early-exit.

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `weight_has_value` | uint32_t (bool) | Whether weight tensor is provided (enables `* weight` step) |
| 1 | `bias_has_value` | uint32_t (bool) | Whether bias tensor is provided (enables `+ bias` step) |
| 2 | `cb_input` | uint32_t | CB index for input tensor (= `c_0`) |
| 3 | `cb_batch_mean` | uint32_t | CB index for batch mean (= `c_1`) |
| 4 | `cb_output_0` | uint32_t | CB index for output tensor (= `c_2`) |
| 5 | `cb_batch_var` | uint32_t | CB index for batch variance (= `c_3`) |
| 6 | `cb_eps` | uint32_t | CB index for epsilon constant (= `c_4`) |
| 7 | `cb_den` | uint32_t | CB index for `1/sqrt(var+eps)` intermediate (= `c_7`) |
| 8 | `cb_weight` | uint32_t | CB index for weight tensor (= `c_5`) |
| 9 | `cb_tmp_1` | uint32_t | CB index for intermediate result (= `c_8`) |
| 10 | `cb_bias` | uint32_t | CB index for bias tensor (= `c_6`) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `num_tiles` | uint32_t | Total number of output tiles this core processes |
| 1 | `tile_freq` | uint32_t | `cHt * cWt` -- tiles per channel, defines broadcast reuse period |
| 2 | `tile_start` | uint32_t | `start_tile_id % cHtWt` -- offset into current channel at start |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `packed_scalar_eps` | uint32_t | Epsilon value packed as float or dual bfloat16 |
| 1 | `src_addr` | uint32_t | Input tensor buffer address |
| 2 | `start_tile_id` | uint32_t | Starting output tile ID for this core |
| 3 | `num_tiles` | uint32_t | Total tiles this core processes |
| 4 | `HtWt` | uint32_t | `cHt * cWt` -- spatial tiles per channel |
| 5 | `n_stride` | uint32_t | Tile stride for batch dimension |
| 6 | `c_stride` | uint32_t | Tile stride for channel dimension |
| 7 | `N` | uint32_t | Number of batches in output |
| 8 | `C` | uint32_t | Number of channels in output |
| 9 | `Ht` | uint32_t | Height in tiles |
| 10 | `Wt` | uint32_t | Width in tiles |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `batch_mean_addr` | uint32_t | Batch mean tensor buffer address |
| 1 | `batch_var_addr` | uint32_t | Batch variance tensor buffer address |
| 2 | `weight_addr` | uint32_t | Weight tensor buffer address (0 if absent) |
| 3 | `bias_addr` | uint32_t | Bias tensor buffer address (0 if absent) |
| 4 | `dst_addr` | uint32_t | Output tensor buffer address |
| 5 | `start_tile_id` | uint32_t | Starting output tile ID for this core |
| 6 | `num_tiles` | uint32_t | Total tiles this core processes |
| 7 | `HtWt` | uint32_t | Spatial tiles per channel |
| 8 | `n_stride` | uint32_t | Tile stride for batch dimension (from mean/var shape) |
| 9 | `c_stride` | uint32_t | Tile stride for channel dimension |
| 10 | `N` | uint32_t | Number of batches in output |
| 11 | `C` | uint32_t | Number of channels in output |
| 12 | `Ht` | uint32_t | Height in tiles |
| 13 | `Wt` | uint32_t | Width in tiles |

## Kernel Implementations

### Compute Kernel -- FPU Path (`batch_norm_kernel.cpp`)

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| compute | RISCV_2 (unpack/math/pack) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2, c_7, c_8 | add, sub, rsqrt, mul (FPU binary ops) |

**Initialization**: `binary_op_init_common(cb_other, cb_bcast, cb_output_0)` configures the FPU for binary eltwise operations.

**Key compute helper calls with exact signatures (FPU path)**:

1. **Denominator computation** (once per channel):
   ```cpp
   // Unpack + compute: var + eps
   add_tiles_init_with_dt(cb_batch_var, cb_eps);  // init FPU add, reconfigures data format if FP32_DEST_ACC_EN
   add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);   // SrcA=cb_batch_var[0], SrcB=cb_eps[0], result -> dst[0]

   // SFPU unary on dst: rsqrt
   rsqrt_tile_init();
   rsqrt_tile(dst0);                               // dst[0] = 1/sqrt(dst[0])

   // Pack result
   pack_tile_with_dt(dst0, cb_den);                // pack dst[0] -> cb_den, reconfigures format if FP32
   ```

2. **Centering** (per tile):
   ```cpp
   sub_tiles_init(cb_other, cb_bcast);             // init FPU subtract
   sub_tiles(cb_other, cb_bcast, 0, 0, 0);         // dst[0] = input[0] - mean[0]
   ```

3. **Normalize** (per tile, chained with centering via dest reuse):
   ```cpp
   binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
   binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
   ```
   This moves the subtraction result from DST to SrcA, loads `cb_den` into SrcB, and multiplies. The key optimization: no need to pack the subtraction result to an intermediate CB and unpack it again. The result remains in DST throughout.

4. **Scale by weight** (per tile, if weight exists):
   ```cpp
   mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
   mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
   ```

5. **Add bias** (per tile, if bias exists):
   ```cpp
   add_tiles_init_with_dt(cb_tmp_1, cb_bias);
   add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
   ```

**CB routing for affine transform**:
```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```
- No affine: normalized result goes directly to `c_2` (output).
- Weight only: normalized -> `c_8` -> multiply -> `c_2`.
- Weight + bias: normalized -> `c_8` -> multiply -> `c_8` -> add bias -> `c_2`.
- Bias only: normalized -> `c_8` -> add bias -> `c_2`.

### Compute Kernel -- SFPU Path (`batch_norm_sfpu_kernel.cpp`)

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| compute | RISCV_2 (unpack/math/pack) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2, c_7, c_8 | copy_tile, add_binary_tile, sub_binary_tile, mul_binary_tile, rsqrt_tile (SFPU binary/unary ops) |

**Initialization**: `unary_op_init_common(cb_other, cb_output_0)` -- note the different init for SFPU path.

**Key compute helper calls with exact signatures (SFPU path)**:

1. **Denominator computation** (once per channel):
   ```cpp
   // Copy var into dst[0]
   copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
   copy_tile(cb_batch_var, 0, 0);        // dst[0] = batch_var

   // Copy eps into dst[1]
   add_binary_tile_init();
   copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
   copy_tile(cb_eps, 0, 1);              // dst[1] = eps
   add_binary_tile(0, 1, 0);             // dst[0] = dst[0] + dst[1]  (var + eps)

   // rsqrt in-place on dst[0]
   rsqrt_tile_init();
   rsqrt_tile(0);                         // dst[0] = 1/sqrt(var + eps)
   pack_tile(0, cb_den);                  // pack to cb_den
   ```

2. **Centering** (per tile):
   ```cpp
   copy_tile_to_dst_init_short_with_dt(cb_bcast, cb_other);
   copy_tile(cb_other, 0, 0);            // dst[0] = input
   sub_binary_tile_init();
   copy_tile_to_dst_init_short_with_dt(cb_other, cb_bcast);
   copy_tile(cb_bcast, 0, 1);            // dst[1] = mean
   sub_binary_tile(0, 1, 0);             // dst[0] = input - mean
   ```

3. **Normalize** (per tile, chained):
   ```cpp
   mul_binary_tile_init();
   copy_tile_to_dst_init_short_with_dt(cb_bcast, cb_den);
   copy_tile(cb_den, 0, 1);              // dst[1] = den = 1/sqrt(var+eps)
   mul_binary_tile(0, 1, 0);             // dst[0] = (input - mean) * den
   pack_tile(0, cb_affine_or_out);
   ```

4. **Scale by weight** (per tile, if weight):
   ```cpp
   copy_tile_to_dst_init_short_with_dt(cb_weight, cb_affine_or_out);
   copy_tile(cb_affine_or_out, 0, 0);    // dst[0] = normalized result
   mul_binary_tile_init();
   copy_tile_to_dst_init_short_with_dt(cb_affine_or_out, cb_weight);
   copy_tile(cb_weight, 0, 1);           // dst[1] = weight
   mul_binary_tile(0, 1, 0);             // dst[0] = result * weight
   pack_tile(0, cb_scaled_output);
   ```

5. **Add bias** (per tile, if bias):
   ```cpp
   copy_tile_to_dst_init_short_with_dt(cb_bias, cb_tmp_1);
   copy_tile(cb_tmp_1, 0, 0);            // dst[0] = scaled result
   add_binary_tile_init();
   copy_tile_to_dst_init_short_with_dt(cb_tmp_1, cb_bias);
   copy_tile(cb_bias, 0, 1);             // dst[1] = bias
   add_binary_tile(0, 1, 0);             // dst[0] = result + bias
   pack_tile(0, cb_output_0);
   ```

**Critical difference from FPU path**: The SFPU path explicitly copies both operands into DST registers (using two DST slots: `i*2` and `i*2+1`) before performing binary operations. This is necessary because SFPU binary ops (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) operate on pairs of DST registers rather than on SrcA/SrcB. The `copy_tile_to_dst_init_short_with_dt` call reconfigures the data format for the source CB, which is essential for FP32 mode.

**SFPU binary op pattern**: `{op}_binary_tile(dst_idx_a, dst_idx_b, dst_idx_result)` -- operands are in DST[dst_idx_a] and DST[dst_idx_b], result stored to DST[dst_idx_result]. The naming convention uses even slots (0, 2, ...) for operand A and odd slots (1, 3, ...) for operand B.

### Reader Kernel (`reader_batch_norm.cpp`)

Provides: input tiles to `c_0`, epsilon constant to `c_4`. Consumes nothing from compute.

### Writer Kernel (`writer_batch_norm.cpp`)

Provides: broadcast mean to `c_1`, broadcast var to `c_3`, broadcast weight to `c_5`, broadcast bias to `c_6`. Consumes: output tiles from `c_2`.

## Implementation Notes

### FPU binary_dest_reuse Optimization

In the FPU kernel, the subtract-then-multiply sequence for normalization uses `binary_dest_reuse_tiles` to avoid an intermediate CB round-trip. After `sub_tiles` computes `(input - mean)` into DST[0], `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` moves DST[0] into SrcA, loads `den` from `c_7` into SrcB, and multiplies. This is more efficient than packing the subtraction result to a CB and unpacking it.

### Two Compute Kernel Variants

The program factory selects between FPU and SFPU kernels at line 288-289:
```cpp
fmt::format("...batch_norm_{}.cpp", fp32_dest_acc_en ? "sfpu_kernel" : "kernel")
```

When `fp32_dest_acc_en = true`:
- All input CBs are configured with `UnpackToDestMode::UnpackToDestFp32` (lines 258-271).
- The SFPU kernel is used because `UnpackToDestFp32` sends data directly to DST registers, bypassing SrcA/SrcB. FPU ops like `add_tiles` cannot work with this mode.
- SFPU provides full 32-bit precision for element-wise operations, whereas FPU is limited to TF32 (19 bits).

### Writer as Parameter Reader

The "writer" kernel does significant reading work: it reads batch_mean, batch_var, weight, and bias from DRAM and broadcasts them. This is a design choice to balance the work between the two data movement RISCs. The "reader" only handles the input tensor and epsilon.

### Broadcast via Scalar Fill

Per-channel parameters are stored as tiles but only the first element matters. The fill functions (`fill_tile_with_first_element_bfloat16`, `fill_tile_with_first_element<float>`) physically write to all 1024 (or 512 packed-pair) positions in L1. This avoids needing hardware broadcast support in the compute path -- the tile is simply a full tile with repeated values, and standard element-wise tile ops work naturally.

### Early Exit for Unused Cores

Cores not in `core_group_1` or `core_group_2` receive all-zero runtime args (line 69-71 in program factory). The compute kernel checks `if (num_tiles == 0) return;` at line 131/164.

### tile_regs Synchronization Pattern

Both kernel variants follow the standard pattern:
1. `tile_regs_acquire()` -- Math core claims DST registers
2. Unpack + compute operations
3. `tile_regs_commit()` -- Hands DST to packer
4. `tile_regs_wait()` -- Packer waits for DST availability
5. `pack_tile` / `pack_tile_with_dt` -- Packs from DST to CB
6. `tile_regs_release()` -- Releases DST for next acquire

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the binary_dest_reuse_tiles pattern in compute kernels? How does DEST_TO_SRCA work and what is the difference between FPU and SFPU compute kernel variants?"
   **Reason**: The FPU kernel uses `binary_dest_reuse_tiles` which is not a standard helper in moreh_common.hpp and needed clarification.
   **Key Findings**: `DEST_TO_SRCA` moves DST contents to SrcA before performing a binary op with SrcB from a CB. This avoids pack/unpack round-trips. SFPU kernels operate directly on DST register pairs and support full FP32 precision. FPU is limited to TF32.

2. **Query**: "How do tile_regs_acquire, tile_regs_commit, tile_regs_wait, and tile_regs_release work?"
   **Reason**: Understanding the DST register synchronization model is critical for analyzing compute kernel correctness and pipelining.
   **Key Findings**: These four functions implement a producer-consumer handoff between the math core (which fills DST) and the packer (which reads DST). `acquire` claims registers and zeroes them. `commit` transfers ownership to packer. `wait` blocks packer until data ready. `release` frees registers for next iteration.

3. **Query**: "What does UnpackToDestMode::UnpackToDestFp32 do versus Default?"
   **Reason**: The program factory configures this mode for all CBs when `fp32_dest_acc_en` is true, and it directly determines which kernel variant is used.
   **Key Findings**: `UnpackToDestFp32` bypasses SrcA/SrcB and unpacks directly to DST in FP32 format. This makes FPU binary ops (which use SrcA/SrcB) incompatible, requiring the SFPU kernel path that operates on DST register pairs.

4. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: The program factory uses this to distribute output tiles across cores.
   **Key Findings**: Divides `units_to_divide` into two groups. Group 1 gets `ceil(total/cores)` tiles, group 2 gets `floor(total/cores)` tiles. Group 1 has `total % cores` members, group 2 has the rest.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Provides the `_with_dt` helper wrappers used in the FPU kernel and utility functions like `pack_tile_with_dt`.
   **Key Information**: `pack_tile_with_dt` calls `pack_reconfig_data_format` before `pack_tile` when `FP32_DEST_ACC_EN` is defined. Similarly, `add_tiles_init_with_dt`, `sub_tiles_init_with_dt`, `mul_tiles_init_with_dt` call `reconfig_data_format` for proper FP32 handling.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Provides the scalar broadcast functions used by the writer kernel.
   **Key Information**: `fill_tile_with_first_element_bfloat16` reads first uint16, packs it as `first_elem << 16 | first_elem`, and writes to all 512 uint32 positions. `fill_tile_with_first_element<float>` reads first float and writes to all 1024 float positions. These are pure L1 memory writes with no compute involvement.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Provides the `create_cb` utility used throughout the program factory.
   **Key Information**: Wraps `CircularBufferConfig` creation. Takes CB index, program, core range, page size, num pages, and data format. Returns `(cb_index, cb_handle)` tuple.

4. **Source**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.hpp`
   **Reason**: Defines the operation's attribute and tensor argument types.
   **Key Information**: `operation_attributes_t` contains `eps`, `memory_config`, `compute_kernel_config`. `tensor_args_t` has `input`, `batch_mean`, `batch_var`, optional `weight`, optional `bias`, optional `output`.
