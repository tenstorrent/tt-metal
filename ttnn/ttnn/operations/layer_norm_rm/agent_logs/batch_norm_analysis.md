# Batch Norm Implementation Analysis (Compute-Core Focus)

## Overview

Batch normalization normalizes each element of an input tensor `x` using pre-computed per-channel mean and variance, with optional affine transformation (gamma/beta). The mathematical operation is:

```
output = gamma * (x - mean) / sqrt(var + eps) + beta
```

**Key difference from layer_norm_rm**: Batch norm receives mean and variance as *pre-computed external tensors* (per-channel), while layer_norm must *compute* mean and variance from the input data itself (per-row). This means batch_norm has no reduction step -- it only performs elementwise broadcast operations. Despite this structural difference, batch_norm provides an excellent reference for:
- CB layout patterns for normalization intermediates
- The `binary_dest_reuse_tiles` optimization for chained operations
- Multi-pass data reuse (broadcast parameters across spatial tiles)
- Scalar/constant CB setup (epsilon)
- Optional affine parameter handling (gamma/beta conditional paths)

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` = total output tiles |
| **Loop structure** | Outer loop over "channel groups" (freq iterations), inner loop over spatial tiles within a channel group. See Multi-Pass Data Reuse section for details. |

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | batch_mean | batch_var | weight (gamma) | bias (beta) | eps |
|----------|-----------|------------|-----------|----------------|-------------|-----|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] | scalar |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW | N/A |
| **Tensor layout** | TILE | TILE | TILE | TILE | TILE | Filled into tile |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | N/A |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM | L1 (CB) |
| **Data type** | BFLOAT16 or FLOAT32 | Same family | Same family | Same family | Same family | Same as input |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Tensor layout** | TILE |
| **Memory layout** | INTERLEAVED |
| **Data type** | Configurable via `dtype` attribute, defaults to input dtype |

### Layout Transformations

No tilize/untilize operations. All tensors are already in TILE layout. The per-channel parameter tensors (mean, var, weight, bias) are broadcast across spatial dimensions (H, W) using `FILL_TILE_WITH_FIRST_ELEMENT` -- the writer kernel reads a single tile from DRAM, extracts the first element, and fills the entire 32x32 tile with that scalar value. This converts a logically [1,C,1,1]-shaped value into a full tile for elementwise operations.

## Data Flow Pattern

### High-Level Flow

```
Reader                          Compute                         Writer
------                          -------                         ------
Read input tile -> CB_0         CB_0 (input)
Fill eps tile   -> CB_4         CB_4 (eps) [persists]
                                                                Read batch_mean -> CB_1 (fill first elem)
                                                                Read batch_var  -> CB_3 (fill first elem)
                                                                Read weight     -> CB_5 (fill first elem, optional)
                                                                Read bias       -> CB_6 (fill first elem, optional)
                                CB_3 + CB_4 -> add -> rsqrt -> CB_7 (den=1/sqrt(var+eps))
                                CB_0 - CB_1 -> sub             [in DST register]
                                DST * CB_7  -> mul -> CB_8 or CB_2 (normalized)
                                CB_8 * CB_5 -> mul -> CB_8 or CB_2 (scaled, if weight)
                                CB_8 + CB_6 -> add -> CB_2     (shifted, if bias)
                                                                CB_2 (output) -> Write to DRAM
```

### Reader Kernel Summary (What It Provides)

The reader (`reader_batch_norm.cpp`) is responsible for:
1. **Epsilon tile (CB_4)**: Fills an entire tile with the epsilon scalar value using `fill_with_val` / `fill_with_val_bfloat16`. This is done **once** at startup and the tile persists in CB_4 for the entire program.
2. **Input tiles (CB_0)**: Reads input tiles one at a time from DRAM in N-C-HW order using TensorAccessor.

### Writer Kernel Summary (What It Consumes/Provides)

The writer (`writer_batch_norm.cpp`) has a dual role -- it acts as both a data provider for per-channel parameters and an output consumer:
1. **Provides** per-channel broadcast tiles: For each new channel, reads mean (CB_1), variance (CB_3), weight (CB_5, optional), and bias (CB_6, optional) tiles from DRAM, then fills each tile with its first element via `FILL_TILE_WITH_FIRST_ELEMENT`.
2. **Consumes** output tiles (CB_2): Waits for compute to produce output tiles, then writes them to DRAM.

**Design note**: The writer reads parameter tensors (mean, var, weight, bias) rather than the reader because the writer uses NOC1 while the reader uses NOC0. This distributes NoC traffic across both networks.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | input_tensor_cb | Input data tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | batch_mean_cb | Per-channel mean (broadcast) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group (persists across HtWt spatial tiles within a channel) |
| c_2 | output_tensor_cb | Final output | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | batch_var_cb | Per-channel variance (broadcast) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group (consumed once per channel for rsqrt computation) |
| c_4 | eps_cb | Epsilon constant (full tile) | 2 tiles | 1 tile | Double | Reader | Compute | Program (filled once, persists entire run) |
| c_5 | weight_cb | Per-channel gamma (broadcast, optional) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group |
| c_6 | bias_cb | Per-channel beta (broadcast, optional) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group |
| c_7 | den_cb | Intermediate: 1/sqrt(var+eps) | 2 tiles | 1 tile | Double | Compute | Compute | Channel group (produced once, reused across HtWt tiles) |
| c_8 | temp_1_cb | Intermediate: scratch for affine ops | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile, used when weight and/or bias exist) |

### CB Assignment Rationale for layer_norm_rm

The batch_norm CB layout provides a template, but layer_norm_rm will need additional CBs for:
- **Reduction scaler**: A CB holding a tile of all 1.0 values (used as the scaler argument to `reduce_tile`)
- **Mean accumulator**: A CB to hold the row-wise sum/mean result
- **Centered values storage**: If computing variance requires a second pass over (x - mean), the centered values must be stored or recomputed
- **Variance accumulator**: A CB to hold the row-wise variance result

## Pipeline Pattern Summary

All CBs have capacity = 2 tiles and block size = 1 tile, so all are **double-buffered**. This allows the producer to fill the next tile while the consumer processes the current tile, enabling overlap between reader/writer and compute.

## Multi-Pass Data Reuse Pattern (Critical for layer_norm_rm)

### The freq/counter Mechanism

The batch_norm compute kernel implements a broadcast reuse pattern controlled by two runtime arguments:

```cpp
// Program factory (line 114-117):
auto counter = start_tile_id % cHtWt;   // offset within channel group
auto freq = cHtWt;                       // tiles per channel group = Ht * Wt

// Compute kernel (line 164-166):
uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
```

**What this means**: `freq` = `Ht * Wt` is the number of spatial tiles per channel. The per-channel parameters (mean, variance, weight, bias, and the computed `den = 1/sqrt(var+eps)`) are loaded **once per channel** and reused across all `Ht * Wt` spatial tiles within that channel.

### How the Reuse Works in the Compute Kernel

Inside `batchnorm_bcast_tiles()`:
1. **Before the inner loop** (once per channel group):
   - Compute `den = rsqrt(batch_var + eps)` from CB_3 and CB_4 -> CB_7
   - Wait on CB_1 (mean), CB_7 (den), CB_5 (weight, if present), CB_6 (bias, if present)

2. **Inner loop** (runs `freq` times, once per spatial tile):
   - Wait for one input tile from CB_0
   - Compute `(input - mean) * den` using the persisted mean/den tiles
   - Optionally multiply by weight, add bias
   - Push result to output CB

3. **After the inner loop** (once per channel group):
   - Pop mean, den, weight, bias tiles -- they are consumed

### Relevance to layer_norm_rm

For layer_norm_rm, the reuse pattern is inverted:
- **Batch norm**: parameters are per-channel, broadcast across spatial (HtWt) tiles
- **Layer norm**: parameters are per-row (per spatial position), computed by reducing across the feature/channel dimension

In layer_norm_rm, the analogous pattern would be:
1. Accumulate mean by reducing across W tiles for each row
2. Store mean, reuse it while subtracting from each input tile in the row
3. Accumulate variance from centered values
4. Compute rsqrt(var + eps), reuse it while normalizing each tile in the row
5. Apply optional gamma/beta

## Compute Kernel Deep Dive

### Two Kernel Variants

The program factory selects between two compute kernels based on `fp32_dest_acc_en`:

1. **`batch_norm_kernel.cpp`** (FPU path, `fp32_dest_acc_en = false`): Uses FPU binary operations (`add_tiles`, `sub_tiles`, `mul_tiles`) and the `binary_dest_reuse_tiles` optimization
2. **`batch_norm_sfpu_kernel.cpp`** (SFPU path, `fp32_dest_acc_en = true`): Uses SFPU operations (`copy_tile` + `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) for full FP32 precision in destination registers

### FPU-Path Compute Kernel (batch_norm_kernel.cpp) -- Primary Reference

#### Initialization

```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```
This initializes the binary operation pipeline, configuring the unpacker and math units for the input/broadcast/output CB data formats.

#### Phase 1: Compute Denominator (1/sqrt(var + eps)) -- Once Per Channel

```cpp
// CB operations:
cb_den_obj.reserve_back(onetile);       // Reserve space in CB_7
cb_batch_var_obj.wait_front(onetile);   // Wait for variance tile in CB_3

// DST register protocol:
tile_regs_acquire();                     // Acquire DST registers for math

// Step 1: var + eps
add_tiles_init_with_dt(cb_batch_var, cb_eps);  // Init add with data format reconfig
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);   // DST[0] = var + eps

// Step 2: rsqrt
rsqrt_tile_init();                       // Init SFPU rsqrt
rsqrt_tile(dst0);                        // DST[0] = 1/sqrt(DST[0])

tile_regs_commit();                      // Hand DST to packer

// Packing:
tile_regs_wait();                        // Packer waits for DST
pack_tile_with_dt(dst0, cb_den);         // Pack DST[0] -> CB_7
tile_regs_release();                     // Release DST for next iteration

cb_batch_var_obj.pop_front(onetile);     // Free variance tile
cb_den_obj.push_back(onetile);           // Signal den tile is ready
```

**Key API signatures used**:
- `add_tiles_init_with_dt(icb0, icb1)`: Inits add with data format reconfig if FP32_DEST_ACC_EN
- `add_tiles(icb0, icb1, itile0, itile1, dst_index)`: Adds tiles from two CBs into DST
- `rsqrt_tile_init()`: Initializes SFPU for reciprocal square root
- `rsqrt_tile(dst_index)`: Computes rsqrt on DST[dst_index] in-place
- `pack_tile_with_dt(dst_index, ocb)`: Packs DST[dst_index] to CB with data format reconfig

#### Phase 2: Normalize Each Spatial Tile -- Inner Loop (freq iterations)

```cpp
// Wait for broadcast parameters (outside inner loop):
cb_bcast_obj.wait_front(onetile);   // mean in CB_1
cb_den_obj.wait_front(onetile);     // den in CB_7

for (uint32_t j = tile_start; j < freq; ++j) {
    cb_other_obj.wait_front(onetile);          // Wait for input tile in CB_0
    cb_affine_or_out_obj.reserve_back(onetile); // Reserve output/intermediate

    tile_regs_acquire();

    // Step 1: input - mean
    sub_tiles_init(cb_other, cb_bcast);
    sub_tiles(cb_other, cb_bcast, 0, 0, 0);    // DST[0] = input - mean

    // Step 2: (input - mean) * den  [DEST REUSE OPTIMIZATION]
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
    // DST[0] (from sub) becomes SRCA, CB_7 tile 0 becomes SRCB
    // Result: DST[0] = (input - mean) * den

    tile_regs_commit();
    tile_regs_wait();
    pack_tile_with_dt(0, cb_affine_or_out);
    tile_regs_release();

    cb_affine_or_out_obj.push_back(onetile);
    cb_other_obj.pop_front(onetile);           // Free input tile (consumed)
    // NOTE: cb_bcast (mean) and cb_den are NOT popped -- they persist for next spatial tile
}
```

**Critical optimization -- `binary_dest_reuse_tiles`**:
- After `sub_tiles`, the result `(input - mean)` is in DST[0]
- Instead of packing to a CB and reading back, `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` keeps DST[0] as SRCA and reads `cb_den` as SRCB
- This avoids one CB write + CB read cycle, saving both memory bandwidth and latency
- **Template parameters**: `EltwiseBinaryType::ELWMUL` = multiply, `EltwiseBinaryReuseDestType::DEST_TO_SRCA` = DST moves to SRCA position

**Full function signature**:
```cpp
template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType binary_reuse_dest>
void binary_dest_reuse_tiles_init(uint32_t icb0);

template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType binary_reuse_dest>
void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index);
```

#### Phase 3: Optional Weight Multiplication

```cpp
if (weight_has_value) {
    cb_scaled_output_obj.reserve_back(onetile);
    cb_affine_or_out_obj.wait_front(1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
    mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_scaled_output);
    tile_regs_release();

    cb_affine_or_out_obj.pop_front(1);
    cb_scaled_output_obj.push_back(onetile);
}
```

#### Phase 4: Optional Bias Addition

```cpp
if (bias_has_value) {
    cb_output_0_obj.reserve_back(onetile);
    cb_tmp_1_obj.wait_front(onetile);

    tile_regs_acquire();
    add_tiles_init_with_dt(cb_tmp_1, cb_bias);
    add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_output_0);
    tile_regs_release();

    cb_tmp_1_obj.pop_front(onetile);
    cb_output_0_obj.push_back(onetile);
}
```

#### Dynamic CB Routing for Optional Affine Parameters

The compute kernel uses a clever pattern to avoid unnecessary intermediate CB writes when weight/bias are absent:

```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```

This means:
- **No weight, no bias**: normalized result goes directly to `cb_output_0` (CB_2)
- **Weight only**: normalized result -> `cb_tmp_1` (CB_8), then weight*result -> `cb_output_0` (CB_2)
- **Bias only**: normalized result -> `cb_tmp_1` (CB_8), then result+bias -> `cb_output_0` (CB_2)
- **Both weight and bias**: normalized result -> `cb_tmp_1` (CB_8), weight*result -> `cb_tmp_1` (CB_8, overwritten), then result+bias -> `cb_output_0` (CB_2)

### SFPU-Path Compute Kernel (batch_norm_sfpu_kernel.cpp)

The SFPU variant performs the same computation but uses different primitives because FP32 destination accumulation requires SFPU operations:

```cpp
// Instead of add_tiles(cb_batch_var, cb_eps, ...):
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
copy_tile(cb_batch_var, i, i * 2);        // Load var to DST[0]
add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
copy_tile(cb_eps, i, i * 2 + 1);          // Load eps to DST[1]
add_binary_tile(i * 2, i * 2 + 1, i * 2); // DST[0] = DST[0] + DST[1]
```

**Key difference**: SFPU operations load both operands into DST registers (`copy_tile` to DST slots) and then perform the operation between DST slots, whereas FPU operations read directly from CB source registers.

**SFPU API signatures used**:
- `copy_tile_to_dst_init_short_with_dt(reconfig_cb, src_cb)`: Init copy with data format reconfig
- `copy_tile(src_cb, src_tile_index, dst_index)`: Copy tile from CB to DST[dst_index]
- `add_binary_tile_init()`: Init SFPU add
- `add_binary_tile(dst_a, dst_b, dst_out)`: DST[dst_out] = DST[dst_a] + DST[dst_b]
- `sub_binary_tile_init()` / `sub_binary_tile(dst_a, dst_b, dst_out)`: Subtraction variant
- `mul_binary_tile_init()` / `mul_binary_tile(dst_a, dst_b, dst_out)`: Multiply variant
- `pack_tile(dst_index, ocb)`: Pack without data format reconfig (FP32 mode)

**Note**: In SFPU mode, the DST register capacity is halved (8 tiles instead of 16, or 4 instead of 8 with double-buffering). The kernel uses DST indices `i*2` and `i*2+1` for operand A and operand B respectively, consuming 2 DST slots per binary operation.

### Initialization Difference

```cpp
// FPU path:
binary_op_init_common(cb_other, cb_bcast, cb_output_0);

// SFPU path:
unary_op_init_common(cb_other, cb_output_0);
```

The SFPU path uses `unary_op_init_common` because it does not use the FPU binary pipeline -- all operations go through the SFPU with explicit `copy_tile` for operand loading.

## Scalar/Constant CB Setup

### Epsilon (CB_4)

The epsilon value is set up by the reader kernel as a **program-lifetime constant**:

```cpp
// Reader kernel (reader_batch_norm.cpp, lines 45-54):
cb_id_eps_obj.reserve_back(onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    float eps_f = 0;
    std::memcpy(&eps_f, &eps, sizeof(float));
    FILL_WITH_VALUE_FLOAT(cb_id_eps, eps_f);   // fill_with_val<1024, float>
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_eps, eps);            // fill_with_val_bfloat16
#endif
cb_id_eps_obj.push_back(onetile);
```

The epsilon scalar is packed into a `uint32_t` in the program factory:
```cpp
// For FLOAT32: std::bit_cast<uint32_t>(scalar)
// For BFLOAT16: pack_two_bfloat16_into_uint32({scalar, scalar})
```

**Compute kernel lifetime**: The compute kernel does `cb_eps_obj.wait_front(onetile)` once at startup and `cb_eps_obj.pop_front(onetile)` once at the very end, ensuring epsilon persists across all channel groups.

### Per-Channel Broadcast Parameters

The writer kernel broadcasts per-channel scalars by:
1. Reading a full tile from DRAM for the channel
2. Calling `FILL_TILE_WITH_FIRST_ELEMENT(cb_id)` which extracts element [0] and fills all 1024 positions

This is necessary because mean, var, weight, and bias are [1,C,1,1] tensors tiled as [1, ceil(C/32), 1, 1] -- each tile contains one channel value in position [0,0] with padding elsewhere.

## Index Calculations

### Program Factory: Tile-to-Channel Mapping

```cpp
// extract_shape_dims returns (N, C, Ht, Wt) where Ht=H/tile_h, Wt=W/tile_w
const auto [aN, aC, aHt, aWt] = extract_shape_dims(input_tensor);

// cHtWt = number of spatial tiles per channel
uint32_t cHtWt = cHt * cWt;

// freq/counter for channel-group iteration:
auto counter = start_tile_id % cHtWt;  // position within current channel group
auto freq = cHtWt;                      // tiles per channel group
```

### Reader: N-C-HW Tile Ordering

```cpp
uint32_t tiles_per_batch = HtWt * C;
uint32_t start_n = start_tile_id / tiles_per_batch;
uint32_t start_remaining = start_tile_id % tiles_per_batch;
uint32_t start_c = start_remaining / HtWt;
uint32_t start_t = start_remaining % HtWt;

// Tile offset with stride computation:
uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;
// n_stride = aHt * aWt * aC * (aN > 1)   -- accounts for padded dimensions
// c_stride = aHt * aWt * (aC > 1)
```

### Compute: Iteration Counting

```cpp
uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
```

This handles the case where a core's tile assignment starts mid-channel-group (tile_start > 0). The first call to `batchnorm_bcast_tiles` processes `tile_freq - tile_start` tiles, then subsequent calls process full channel groups of `tile_freq` tiles.

## Memory Access Patterns

### Read Pattern
- **Input tiles**: Sequential within a channel (contiguous spatial tiles), with stride jumps at channel and batch boundaries
- **Per-channel parameters**: One tile read per channel, then broadcast (same tile reused for all HtWt spatial tiles in that channel)
- **Epsilon**: Single tile, read once

### Write Pattern
- **Output tiles**: Sequential, written one tile at a time using TensorAccessor with sequential `start_tile_id + num_tiles_written` page IDs

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (row-major iteration) |
| **Grid dimensions** | compute_with_storage_grid_size.x * compute_with_storage_grid_size.y |
| **Total cores** | num_cores_x * num_cores_y |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split via `split_work_to_cores` -- group 1 gets `ceil(total/cores)` tiles, group 2 gets `floor(total/cores)` tiles |

The `split_work_to_cores` utility divides total output tiles across all available cores. Cores not in either group receive zero-filled runtime args and return immediately (`if (num_tiles == 0) return;`).

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t (bool) | Whether gamma (weight) tensor is provided |
| 1 | bias_has_value | uint32_t (bool) | Whether beta (bias) tensor is provided |
| 2 | cb_input | uint32_t | CB index for input tensor (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon constant (c_4) |
| 7 | cb_den | uint32_t | CB index for 1/sqrt(var+eps) intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for gamma tensor (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for scratch intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB index for beta tensor (c_6) |

**Design note**: CB indices are passed as compile-time args rather than hardcoded, enabling flexible CB assignment. This is a good pattern to follow for layer_norm_rm.

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles this core must process |
| 1 | freq (tile_freq) | uint32_t | Tiles per channel group (Ht * Wt) -- controls when per-channel params are refreshed |
| 2 | counter (tile_start) | uint32_t | Starting offset within a channel group (for partial first group) |

### ComputeConfig Settings

```cpp
ComputeConfig{
    .fp32_dest_acc_en = fp32_dest_acc_en,
    .unpack_to_dest_mode = unpack_to_dest_mode,  // per-CB FP32 unpack settings
    .compile_args = compute_kernel_args
}
```

When `fp32_dest_acc_en = true`:
- All input/intermediate/output CBs get `UnpackToDestMode::UnpackToDestFp32`
- The SFPU kernel variant is selected
- DST register capacity is halved (8 tiles total, 4 per half with double-buffering)

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_batch_norm.cpp | RISCV_0 | NOC0 | DRAM (input) | CB_0 (input), CB_4 (eps) | Read input tiles, fill eps tile |
| batch_norm_kernel.cpp (FPU) | RISCV_2 | N/A | CB_0, CB_1, CB_3, CB_4, CB_5, CB_6 | CB_2, CB_7, CB_8 | add, sub, mul, rsqrt, binary_dest_reuse |
| batch_norm_sfpu_kernel.cpp (SFPU) | RISCV_2 | N/A | CB_0, CB_1, CB_3, CB_4, CB_5, CB_6 | CB_2, CB_7, CB_8 | copy_tile, add/sub/mul_binary_tile, rsqrt |
| writer_batch_norm.cpp | RISCV_1 | NOC1 | DRAM (mean, var, weight, bias), CB_2 | CB_1, CB_3, CB_5, CB_6, DRAM (output) | Read params, fill_tile_with_first_element, write output |

### Compute Kernel Key Logic

**FPU path** (`batch_norm_kernel.cpp`):
- Uses `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` to fuse subtract and multiply without intermediate CB write
- Uses moreh_common.hpp helpers: `add_tiles_init_with_dt`, `sub_tiles_init`, `mul_tiles_init_with_dt`, `pack_tile_with_dt`
- All operations use 1 DST register slot (dst0 = 0)

**SFPU path** (`batch_norm_sfpu_kernel.cpp`):
- Uses paired DST slots: operand A at `i*2`, operand B at `i*2+1`
- Explicit `copy_tile` to load each operand into DST before binary SFPU operations
- Uses `pack_tile` (not `pack_tile_with_dt`) since FP32 dest is always active in this path

## Implementation Notes

### Patterns Directly Applicable to layer_norm_rm

1. **Epsilon CB as program-lifetime constant**: Fill once in reader, consume at start/end in compute. For layer_norm_rm, also need a "scaler" CB filled with 1.0 for `reduce_tile` (or 1/W for mean computation).

2. **`binary_dest_reuse_tiles` for chained operations**: In layer_norm_rm, the sequence `(x - mean) * rsqrt_var` can use this same optimization to avoid an intermediate CB write between subtract and multiply.

3. **Dynamic CB routing for optional parameters**: The `cb_affine_or_out` / `cb_scaled_output` pattern for conditional weight/bias is directly reusable.

4. **Per-channel parameter broadcast via `FILL_TILE_WITH_FIRST_ELEMENT`**: For layer_norm_rm, gamma and beta are per-feature (applied along the last dimension), so they may need `fill_tile_with_first_row` instead if operating on row-major data, or may be read as full tiles if the feature dimension maps to tile columns.

5. **Two-variant kernel selection** (FPU vs SFPU): Follow the same `fp32_dest_acc_en` conditional pattern.

### Key Differences for layer_norm_rm

1. **Reduction required**: Batch norm receives pre-computed statistics; layer_norm must compute mean and variance using `reduce_tile` with `PoolType::SUM` and `ReduceDim::REDUCE_ROW`.

2. **Two-pass or fused computation**: Layer norm needs either:
   - Two passes over the data (pass 1: compute mean; pass 2: compute variance and normalize), or
   - Welford's online algorithm (single pass for mean+variance), or
   - Compute E[x] and E[x^2] in one pass, then var = E[x^2] - E[x]^2

3. **Row-major layout**: The input is row-major, not tiled, which affects how tiles are read and how reduction maps to the row dimension.

4. **Reduction scaler CB**: `reduce_tile` requires a scaler CB. For computing mean, fill with `1/W` (where W is the row width). For computing sum, fill with `1.0`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does binary_dest_reuse_tiles work in the compute kernel API?"
   **Reason**: The FPU-path kernel uses this optimization to chain sub_tiles and mul_tiles without intermediate CB write
   **Key Findings**: DEST_TO_SRCA moves current DST content to SRCA position, reads the CB operand as SRCB, performs the binary op, and writes result back to DST. This eliminates one CB write+read cycle per tile in the normalization inner loop.

2. **Query**: "What is the tile_regs_acquire / commit / wait / release protocol?"
   **Reason**: Every compute phase in the kernel uses this four-step protocol
   **Key Findings**: acquire gives DST to math core, commit hands to pack core, wait blocks pack until math is done, release frees DST for next iteration. Enables pipelined math+pack execution. With fp32_dest_acc_en=true, DST capacity halves from 16 to 8 tiles (4 per half with double-buffering).

3. **Query**: "How does the experimental::CircularBuffer API work?"
   **Reason**: Both kernel variants use this modern C++ wrapper instead of legacy cb_* functions
   **Key Findings**: Object-oriented wrapper that encapsulates CB ID. Methods: reserve_back(n), push_back(n), wait_front(n), pop_front(n), plus non-blocking variants pages_reservable_at_back(n) and pages_available_at_front(n). Functionally identical to legacy API.

4. **Query**: "How does reduce_tile work for row-wise reduction?"
   **Reason**: layer_norm_rm will need reduce_tile for computing mean and variance
   **Key Findings**: `reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(icb, icb_scaler, ocb)` + `reduce_tile(icb, icb_scaler, itile, itile_scaler, idst)`. Sums across columns of a 32x32 tile producing a column vector. Requires scaler CB (1.0 for SUM). Must call `reduce_uninit()` after all reductions. acquire_dst before, release_dst after.

5. **Query**: "How does UnpackToDestMode::UnpackToDestFp32 affect DST capacity?"
   **Reason**: The batch_norm kernel selects SFPU variant when fp32_dest_acc_en is true
   **Key Findings**: FP32 mode halves DST capacity. With double-buffering (default): 4 tiles available (vs 8 in bf16 mode). Without double-buffering (dst_full_sync_en=true): 8 tiles available. Important constraint: CBs configured with UnpackToDestFp32 cannot be used in SrcA/SrcB operations directly.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Compute kernel includes this header for `_with_dt` helper functions
   **Key Information**: Provides `add_tiles_init_with_dt`, `mul_tiles_init_with_dt`, `sub_tiles_init_with_dt`, `pack_tile_with_dt`, `copy_tile_init_with_dt` -- all of which add `reconfig_data_format` calls when `FP32_DEST_ACC_EN` is defined. Also provides composite helpers like `mul_tiles_to_cb`, `add_tiles_to_cb`, `sub_tiles_to_cb` that handle the full CB wait/acquire/compute/commit/wait/pack/release/pop/push cycle.

2. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Program factory uses `create_cb` utility
   **Key Information**: `create_cb(cb_index, program, core_spec, page_size, num_pages, data_format)` returns `tuple<uint32_t, CBHandle>`. Supports single CB or array of CB indices sharing the same config. Also provides `calculate_total_cb_size()` for L1 budget checking.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Writer kernel uses `FILL_TILE_WITH_FIRST_ELEMENT` for per-channel broadcast
   **Key Information**: `fill_tile_with_first_element_bfloat16(cb_id)` reads first uint16 element, packs it into uint32, fills all 512 uint32 positions. Float32 variant `fill_tile_with_first_element<float>(cb_id)` fills all 1024 float positions. Also provides `fill_with_val_bfloat16(cb_id, packed_scalar)` for uniform scalar fill, and row/column broadcast variants (`fill_tile_with_first_row`, `fill_tile_with_first_column`).

4. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: Exact function signatures for binary_dest_reuse_tiles
   **Key Information**: Template parameters are `<EltwiseBinaryType, EltwiseBinaryReuseDestType>`. Init takes `(uint32_t icb0)`. Execute takes `(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)`. The init configures unpack A with the reuse type flag, which controls whether DST content moves to SRCA or SRCB position.
