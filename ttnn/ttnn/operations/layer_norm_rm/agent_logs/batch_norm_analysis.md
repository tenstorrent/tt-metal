# Batch Norm Implementation Analysis

## Overview

Batch normalization normalizes each channel across the batch dimension using pre-computed batch mean and batch variance statistics, then optionally applies an affine transformation using learnable gamma (weight) and beta (bias) parameters. The formula is:

```
output = (input - batch_mean) / sqrt(batch_var + eps) * weight + bias
```

The mean and variance are NOT computed in this kernel -- they are provided as pre-computed input tensors. This is a key distinction from layer normalization, where the statistics must be computed inline.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `output.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Flat tile iteration with channel-frequency grouping (see Multi-Pass section) |

Each tile of the output tensor is one unit of work. Tiles are distributed across cores using `split_work_to_cores`.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor | Batch Mean | Batch Var | Weight (optional) | Bias (optional) |
|----------|--------------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [N, C, 1, 1] or [1, C, 1, 1] | [N, C, 1, 1] or [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Configurable via `operation_attributes.dtype` |

### Layout Transformations

- **Batch mean, batch var, weight, bias**: These are per-channel scalars stored in tiles. The writer kernel reads one tile per channel and calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the first element of the tile across the entire 32x32 tile. This effectively creates a uniform tile that can be used in element-wise operations with each input tile in that channel.
- **Epsilon**: The reader kernel fills an entire tile with the epsilon scalar value using `fill_with_val<1024, float>` (FP32) or `fill_with_val_bfloat16` (BF16). This creates a uniform epsilon tile that persists for the entire program.

## Data Flow Pattern

The data flow has a notable design: the **reader** handles input data and epsilon, while the **writer** handles batch_mean, batch_var, weight, bias, AND output writes. This is a **split reader** pattern where read responsibilities are divided between the reader (input + eps) and writer (statistics + affine parameters) kernels.

### Step-by-step flow:

1. **Reader**: Fills epsilon CB (c_4) once with scalar eps value, pushes 1 tile. This tile persists for the entire program.
2. **Writer**: For each channel group, reads one tile each of batch_mean, batch_var, (optionally weight, bias) from DRAM. Applies `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the per-channel scalar across the full tile. Pushes to CB c_1, c_3, (c_5, c_6).
3. **Reader**: For each input tile in the channel group, reads one input tile from DRAM and pushes to CB c_0.
4. **Compute**: Processes tiles through the normalization pipeline (see Compute Kernel section).
5. **Writer**: Waits for each output tile in CB c_2, writes it to DRAM output buffer.
6. Steps 2-5 repeat for each channel across all batches.

### Critical insight -- Channel-frequency grouping

The batch_mean/var/weight/bias tiles are **reused across all spatial tiles (HtxWt) within a channel**. The compute kernel uses a frequency/counter mechanism (`tile_freq` = `cHt * cWt`, `tile_start` = `start_tile_id % cHtWt`) to know when a new channel begins and new statistics tiles are needed.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tensor tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | cb_batch_mean | Broadcast batch mean | 2 tiles | 1 tile | Double | Writer | Compute | Block (per channel group) |
| c_2 | cb_output | Output tensor tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | cb_batch_var | Broadcast batch variance | 2 tiles | 1 tile | Double | Writer | Compute | Block (per channel group) |
| c_4 | cb_eps | Epsilon scalar tile | 2 tiles | 1 tile | Double | Reader | Compute | **Program** (entire kernel) |
| c_5 | cb_weight | Broadcast gamma/weight | 2 tiles | 1 tile | Double | Writer | Compute | Block (per channel group) |
| c_6 | cb_bias | Broadcast beta/bias | 2 tiles | 1 tile | Double | Writer | Compute | Block (per channel group) |
| c_7 | cb_den | Intermediate: `1/sqrt(var+eps)` | 2 tiles | 1 tile | Double | Compute | Compute | Block (per channel group) |
| c_8 | cb_tmp_1 | Intermediate: normalized or scaled result | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile) |

### Multi-pass data reuse patterns (CBs that persist across phases)

**c_4 (epsilon)**: Pushed once by the reader at program start, waited on by compute at the start, and popped at the very end of the compute kernel (`cb_pop_front(cb_eps, onetile)` at line 190 of the non-SFPU kernel). This means epsilon occupies its CB for the entire duration of the program. The compute kernel does `cb_wait_front(cb_eps, onetile)` once before the main loop and `cb_pop_front(cb_eps, onetile)` after all iterations.

**c_1 (batch_mean), c_3 (batch_var), c_5 (weight), c_6 (bias)**: These persist for one channel group (all HtxWt spatial tiles within one N,C pair). They are waited on at the start of each `batchnorm_bcast_tiles` call and popped at the end of that call. Within the call, the inner loop processes `freq - tile_start` input tiles using the same batch_mean/var/weight/bias tiles.

**c_7 (den = 1/sqrt(var+eps))**: Computed once per channel group from batch_var + eps. Persists for all spatial tiles in that channel group, then popped.

## Pipeline Pattern Summary

All CBs have capacity = 2 tiles and block size = 1 tile, making them all **double-buffered**. This allows the producer to fill the next slot while the consumer processes the current slot, enabling overlap between data movement and compute.

## Index Calculations

### Tile indexing in the program factory

The output tensor is conceptualized as a flat sequence of tiles in `[N, C, Ht, Wt]` order. The key derived values are:

- `cHtWt = cHt * cWt`: tiles per spatial plane per channel
- `start_tile_id`: the global starting tile index for this core
- `counter = start_tile_id % cHtWt`: offset within the current channel's spatial plane
- `freq = cHtWt`: number of spatial tiles per channel

### How the compute kernel uses freq/counter

```
complete_iterations = (num_tiles + tile_start) / tile_freq
remaining_iterations = (num_tiles + tile_start) % tile_freq
```

Each `complete_iterations` call to `batchnorm_bcast_tiles` processes one full channel group. The first call may start mid-channel (at `tile_start`). After the first call, `tile_start` resets to 0. The remaining iterations handle the last partial channel.

### Reader/Writer tile offset calculation

Both reader and writer compute tile offsets using stride-based indexing:
- `tile_offset = start_n * n_stride + start_c * c_stride + start_t`
- `n_stride = aHt * aWt * aC * (aN > 1)` (batch stride, 0 if single batch)
- `c_stride = aHt * aWt * (aC > 1)` (channel stride, 0 if single channel)

For the writer's batch_mean/var/weight/bias, the same offset scheme applies but uses the `b` tensor shapes.

## Memory Access Patterns

### Read Pattern

**Input (Reader)**: Sequential within a channel's spatial plane, with stride jumps between channels and batches. The inner loop iterates `t` from `start_t` to `HtWt`, reading tiles sequentially. At channel boundaries, adds `next_channel_shift = c_stride - HtWt`. At batch boundaries, adds `next_batch_shift = n_stride - c_stride * C`.

**Batch mean/var/weight/bias (Writer)**: One tile per channel, read once per channel group. The tile offset advances by `c_stride` per channel and `next_batch_shift` per batch. After reading, `FILL_TILE_WITH_FIRST_ELEMENT` broadcasts the scalar across the tile in-place in L1.

**Epsilon (Reader)**: One tile, written once at the start. No subsequent DRAM reads.

### Write Pattern

**Output (Writer)**: Sequential tile writes using `noc_async_write_tile(start_tile_id + num_tiles_written, dst, ...)`. The write pattern is linear across the output tile space.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (used as 1D via row-major enumeration) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores_x * num_cores_y` |
| **Work per core** | `num_output_tiles / num_cores` tiles (with remainder handling) |
| **Load balancing** | Two-group split: `core_group_1` gets `num_tiles_per_core_group_1`, `core_group_2` gets `num_tiles_per_core_group_2` (differs by at most 1) |

Uses `split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major=true)` which returns:
- `num_cores`: actual number of cores used
- `all_cores`: all active cores
- `core_group_1`, `core_group_2`: two groups for even/remainder distribution
- `num_tiles_per_core_group_1`, `num_tiles_per_core_group_2`: tiles per core for each group

Cores not in either group receive zeroed-out runtime arguments and immediately return.

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if gamma/weight tensor is provided, 0 otherwise |
| 1 | bias_has_value | uint32_t | 1 if beta/bias tensor is provided, 0 otherwise |
| 2 | cb_input | uint32_t | CB index for input tensor (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon (c_4) |
| 7 | cb_den | uint32_t | CB index for 1/sqrt(var+eps) intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for weight/gamma (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for temporary intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB index for bias/beta (c_6) |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src | uint32_t | CB index for input (c_0) |
| 1 | cb_id_eps | uint32_t | CB index for epsilon (c_4) |
| 2+ | TensorAccessorArgs | varies | Tensor accessor parameters for input tensor |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | Whether weight tensor is present |
| 1 | bias_has_value | uint32_t | Whether bias tensor is present |
| 2 | cb_id_src | uint32_t | CB index for batch_mean (c_1) |
| 3 | cb_id_dst | uint32_t | CB index for output (c_2) |
| 4 | cb_id_batch_var | uint32_t | CB index for batch_var (c_3) |
| 5 | cb_id_weight | uint32_t | CB index for weight (c_5) |
| 6 | cb_id_bias | uint32_t | CB index for bias (c_6) |
| 7+ | TensorAccessorArgs | varies | Tensor accessor params for batch_mean, output, batch_var, weight, bias (chained) |

### Runtime Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total number of output tiles this core processes |
| 1 | freq (tile_freq) | uint32_t | Number of spatial tiles per channel group (`cHt * cWt`) |
| 2 | counter (tile_start) | uint32_t | Starting offset within the first channel group (`start_tile_id % cHtWt`) |

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value packed as float32 bits or two bfloat16 |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | Starting tile index for this core |
| 3 | num_tiles | uint32_t | Number of tiles to process |
| 4 | HtWt (cHtWt) | uint32_t | Spatial tiles per channel |
| 5 | n_stride | uint32_t | Tile stride between batches (0 if single batch) |
| 6 | c_stride | uint32_t | Tile stride between channels (0 if single channel) |
| 7 | N (cN) | uint32_t | Number of batches |
| 8 | C (cC) | uint32_t | Number of channels |
| 9 | Ht (cHt) | uint32_t | Height in tiles (unused directly by reader, part of args) |
| 10 | Wt (cWt) | uint32_t | Width in tiles (unused directly by reader, part of args) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Batch mean tensor buffer address |
| 1 | batch_var_addr | uint32_t | Batch variance tensor buffer address |
| 2 | weight_addr | uint32_t | Weight tensor address (0 if not present) |
| 3 | bias_addr | uint32_t | Bias tensor address (0 if not present) |
| 4 | dst_addr | uint32_t | Output tensor buffer address |
| 5 | start_tile_id | uint32_t | Starting tile index for this core |
| 6 | num_tiles | uint32_t | Number of tiles to process |
| 7 | HtWt | uint32_t | Spatial tiles per channel |
| 8 | n_stride | uint32_t | Batch stride for mean/var/weight/bias tensors |
| 9 | c_stride | uint32_t | Channel stride for mean/var/weight/bias tensors |
| 10 | N | uint32_t | Number of batches |
| 11 | C | uint32_t | Number of channels |
| 12 | Ht | uint32_t | Height in tiles |
| 13 | Wt | uint32_t | Width in tiles |

## Kernel Implementations

### Compute Kernel -- FPU Path (batch_norm_kernel.cpp)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`

Selected when `fp32_dest_acc_en == false`. Uses FPU-based binary operations that operate directly with CB-to-SrcA/SrcB unpacking.

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute (FPU) | RISCV_2 (Unpack+Math+Pack) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2, c_7, c_8 | add_tiles, sub_tiles, rsqrt_tile, binary_dest_reuse_tiles (mul), mul_tiles, pack_tile_with_dt |

#### Initialization

```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```

Signature: `binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb)` -- initializes unpack, math, and pack hardware for binary operations.

#### Compute flow per channel group (`batchnorm_bcast_tiles`)

**Phase 1: Compute denominator (once per channel group)**

```cpp
// batch_var + eps -> rsqrt -> den = 1/sqrt(batch_var + eps)
cb_reserve_back(cb_den, onetile);
cb_wait_front(cb_batch_var, onetile);

tile_regs_acquire();
add_tiles_init_with_dt(cb_batch_var, cb_eps);     // FPU init for add
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);       // dst0 = batch_var[0] + eps[0]
rsqrt_tile_init();                                  // SFPU init for rsqrt
rsqrt_tile(dst0);                                   // dst0 = 1/sqrt(dst0)
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(dst0, cb_den);                    // pack to intermediate CB c_7
tile_regs_release();

cb_pop_front(cb_batch_var, onetile);
cb_push_back(cb_den, onetile);
```

Key signatures:
- `add_tiles_init_with_dt(uint32_t icb0, uint32_t icb1)` -- reconfigures data format and inits FPU add
- `add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` -- FPU element-wise add from two CBs to DST
- `rsqrt_tile_init()` -- SFPU init for reciprocal square root
- `rsqrt_tile(uint32_t idst)` -- SFPU rsqrt on DST register in-place
- `pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb)` -- reconfigures pack format, packs DST to CB

**Phase 2: Wait for broadcast tiles**

```cpp
cb_wait_front(cb_bcast, onetile);     // batch_mean
cb_wait_front(cb_den, onetile);       // 1/sqrt(var+eps)
if (weight_has_value) cb_wait_front(cb_weight, onetile);
if (bias_has_value) cb_wait_front(cb_bias, onetile);
```

These tiles persist for the entire inner loop over spatial tiles.

**Phase 3: Per-tile normalization (inner loop, runs `freq - tile_start` times)**

The output destination CB is dynamically chosen based on which affine parameters are present:
```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```

This routing avoids unnecessary intermediate writes:
- No affine: normalized result goes directly to `cb_output_0`
- Weight only: intermediate goes to `cb_tmp_1`, then weight*intermediate to `cb_output_0`
- Bias only: intermediate goes to `cb_tmp_1`, then intermediate+bias to `cb_output_0`
- Both: intermediate to `cb_tmp_1`, weight*intermediate to `cb_tmp_1`, then +bias to `cb_output_0`

**Step 3a: Subtract mean**

```cpp
cb_wait_front(cb_other, onetile);    // wait for input tile
cb_reserve_back(cb_affine_or_out, onetile);

tile_regs_acquire();
sub_tiles_init(cb_other, cb_bcast);
sub_tiles(cb_other, cb_bcast, 0, 0, 0);   // dst0 = input - batch_mean
```

Signature: `sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` -- FPU element-wise subtract.

**Step 3b: Multiply by denominator (dest reuse pattern)**

```cpp
binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
```

This is a critical optimization. Instead of packing the subtraction result to a CB and then unpacking it again for multiplication, `binary_dest_reuse_tiles` keeps the result in DST and moves it to SrcA, then unpacks `cb_den` to SrcB, and performs the multiply. The result overwrites DST[0].

Signature: `binary_dest_reuse_tiles<EltwiseBinaryType, EltwiseBinaryReuseDestType>(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)` -- performs binary op where one operand comes from DST (moved to SrcA or SrcB) and the other from a CB.

Template parameters:
- `EltwiseBinaryType::ELWMUL` -- multiply operation
- `EltwiseBinaryReuseDestType::DEST_TO_SRCA` -- the current DST value is moved to SrcA, the CB tile goes to SrcB

```cpp
tile_regs_commit();
tile_regs_wait();
pack_tile_with_dt(0, cb_affine_or_out);
tile_regs_release();
cb_push_back(cb_affine_or_out, onetile);
cb_pop_front(cb_other, onetile);
```

**Step 3c: Multiply by weight (optional)**

```cpp
if (weight_has_value) {
    cb_reserve_back(cb_scaled_output, onetile);
    cb_wait_front(cb_affine_or_out, 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
    mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);  // dst0 = normalized * weight
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_scaled_output);
    tile_regs_release();

    cb_pop_front(cb_affine_or_out, 1);
    cb_push_back(cb_scaled_output, onetile);
}
```

Signature: `mul_tiles_init_with_dt(uint32_t icb0, uint32_t icb1)` -- reconfigures data format and inits FPU multiply.

**Step 3d: Add bias (optional)**

```cpp
if (bias_has_value) {
    cb_reserve_back(cb_output_0, onetile);
    cb_wait_front(cb_tmp_1, onetile);

    tile_regs_acquire();
    add_tiles_init_with_dt(cb_tmp_1, cb_bias);
    add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);   // dst0 = scaled + bias
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_output_0);
    tile_regs_release();

    cb_pop_front(cb_tmp_1, onetile);
    cb_push_back(cb_output_0, onetile);
}
```

**Phase 4: Pop broadcast tiles**

```cpp
cb_pop_front(cb_bcast, onetile);     // release batch_mean
cb_pop_front(cb_den, onetile);       // release denominator
if (weight_has_value) cb_pop_front(cb_weight, onetile);
if (bias_has_value) cb_pop_front(cb_bias, onetile);
```

### Compute Kernel -- SFPU Path (batch_norm_sfpu_kernel.cpp)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`

Selected when `fp32_dest_acc_en == true`. Uses SFPU-based binary operations that operate entirely within DST registers.

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute (SFPU) | RISCV_2 (Unpack+Math+Pack) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2, c_7, c_8 | copy_tile, add_binary_tile, sub_binary_tile, mul_binary_tile, rsqrt_tile, pack_tile |

#### Key difference from FPU path

When FP32 accumulation is enabled, `UnpackToDestMode::UnpackToDestFp32` is set for all relevant CBs. This mode unpacks tiles directly into DST registers (instead of SrcA/SrcB). Arithmetic is then performed using SFPU binary operations that operate on pairs of DST register slots.

The SFPU path uses a different register layout convention: each operand occupies a DST slot at `i * 2` and `i * 2 + 1` respectively:
```cpp
copy_tile(cb_batch_var, i, i * 2);     // operand A at DST[0]
copy_tile(cb_eps, i, i * 2 + 1);       // operand B at DST[1]
add_binary_tile(i * 2, i * 2 + 1, i * 2);  // DST[0] = DST[0] + DST[1]
```

#### SFPU binary operation signatures

- `copy_tile_to_dst_init_short_with_dt(uint32_t icb_format, uint32_t icb)` -- reconfigures data format for copy
- `copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)` -- unpacks tile from CB directly into DST[idst]
- `add_binary_tile_init()` -- SFPU init for addition
- `add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` -- DST[odst] = DST[idst0] + DST[idst1]
- `sub_binary_tile_init()` -- SFPU init for subtraction
- `sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` -- DST[odst] = DST[idst0] - DST[idst1]
- `mul_binary_tile_init()` -- SFPU init for multiplication
- `mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` -- DST[odst] = DST[idst0] * DST[idst1]
- `rsqrt_tile_init()` -- SFPU init for reciprocal square root
- `rsqrt_tile(uint32_t idst)` -- DST[idst] = 1/sqrt(DST[idst])
- `pack_tile(uint32_t ifrom_dst, uint32_t icb)` -- packs DST register to CB

#### Initialization

```cpp
unary_op_init_common(cb_other, cb_output_0);
```

Note: The SFPU path uses `unary_op_init_common` instead of `binary_op_init_common` because the SFPU binary operations work within DST registers, not via the FPU's SrcA/SrcB path.

#### SFPU Phase 1: Compute denominator

```cpp
tile_regs_acquire();
tile_regs_wait();
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
copy_tile(cb_batch_var, 0, 0);      // DST[0] = batch_var
add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
copy_tile(cb_eps, 0, 1);            // DST[1] = eps
add_binary_tile(0, 1, 0);           // DST[0] = batch_var + eps
rsqrt_tile_init();
rsqrt_tile(0);                       // DST[0] = 1/sqrt(batch_var + eps)
pack_tile(0, cb_den);               // pack to CB c_7
tile_regs_commit();
tile_regs_release();
```

#### SFPU Phase 3: Per-tile normalization

```cpp
// Step 3a: input - batch_mean
copy_tile(cb_other, 0, 0);          // DST[0] = input
sub_binary_tile_init();
copy_tile(cb_bcast, 0, 1);          // DST[1] = batch_mean
sub_binary_tile(0, 1, 0);           // DST[0] = input - batch_mean

// Step 3b: multiply by denominator
mul_binary_tile_init();
copy_tile(cb_den, 0, 1);            // DST[1] = den
mul_binary_tile(0, 1, 0);           // DST[0] = (input - mean) * den
pack_tile(0, cb_affine_or_out);     // pack result

// Step 3c: multiply by weight (if present)
copy_tile(cb_affine_or_out, 0, 0);  // DST[0] = normalized
mul_binary_tile_init();
copy_tile(cb_weight, 0, 1);         // DST[1] = weight
mul_binary_tile(0, 1, 0);           // DST[0] = normalized * weight
pack_tile(0, cb_scaled_output);

// Step 3d: add bias (if present)
copy_tile(cb_tmp_1, 0, 0);          // DST[0] = scaled
add_binary_tile_init();
copy_tile(cb_bias, 0, 1);           // DST[1] = bias
add_binary_tile(0, 1, 0);           // DST[0] = scaled + bias
pack_tile(0, cb_output_0);
```

### Reader Kernel (Summary)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp`

- Provides: input tiles to c_0, epsilon tile to c_4
- Fills epsilon tile once at start using `fill_with_val<1024, float>` (FP32) or `fill_with_val_bfloat16` (BF16)
- Reads input tiles sequentially within channel spatial planes, with stride jumps between channels/batches
- Uses TensorAccessor for input tensor DRAM reads

### Writer Kernel (Summary)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`

- Provides: batch_mean to c_1, batch_var to c_3, weight to c_5, bias to c_6
- Consumes: output tiles from c_2
- For each channel group: reads one tile each of mean/var/weight/bias, applies `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast scalar, pushes to CBs
- Then loops over spatial tiles writing output from c_2 to DRAM
- Uses TensorAccessor for all tensor DRAM reads/writes

## Implementation Notes

### Two compute kernel variants

The program factory selects the kernel variant at compile time based on `fp32_dest_acc_en`:
- `batch_norm_kernel.cpp` (FPU path): Standard precision, uses FPU binary ops (add_tiles, sub_tiles, mul_tiles) and the `binary_dest_reuse_tiles` optimization
- `batch_norm_sfpu_kernel.cpp` (SFPU path): FP32 accumulation, uses SFPU binary ops (copy_tile + add/sub/mul_binary_tile) that operate entirely in DST registers

### binary_dest_reuse_tiles optimization (FPU path only)

The FPU path chains subtraction and multiplication without an intermediate pack/unpack cycle. After `sub_tiles` leaves the result in DST[0], `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` moves DST[0] to SrcA, unpacks the denominator from `cb_den` to SrcB, and performs the multiply. This saves one pack+unpack cycle per tile, which is significant since it happens for every spatial tile.

### FILL_TILE_WITH_FIRST_ELEMENT broadcast mechanism

The per-channel statistics (mean, var, weight, bias) are stored as 1x1 scalars within tiles. After reading from DRAM, the writer calls `FILL_TILE_WITH_FIRST_ELEMENT` which reads element [0] and writes it to all 1024 positions in the tile. This converts a scalar-in-tile to a uniform tile, enabling standard element-wise tile operations in the compute kernel.

### Dynamic CB routing for affine parameters

The compute kernel dynamically routes intermediate results through different CBs depending on which affine parameters are present. The logic ensures that the final result always ends up in `cb_output_0` (c_2):
- `cb_affine_or_out`: where normalized result goes (tmp if affine needed, output if not)
- `cb_scaled_output`: where weight-scaled result goes (tmp if bias needed, output if not)

### UnpackToDestMode::UnpackToDestFp32

When FP32 accumulation is enabled, all input CBs are configured with `UnpackToDestMode::UnpackToDestFp32`. This changes the unpack behavior so that data goes directly from L1 to DST registers (bypassing SrcA/SrcB), enabling SFPU-based computation with full FP32 precision in the accumulator.

### Epsilon handling

Epsilon is packed differently based on data type:
- FP32: `std::bit_cast<uint32_t>(scalar)` -- direct bit representation
- BF16: `pack_two_bfloat16_into_uint32({scalar, scalar})` -- two copies packed into one uint32

The reader unpacks this and fills the tile appropriately.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What is the binary_dest_reuse_tiles API in compute kernels?"
   **Reason**: Understanding the dest reuse optimization pattern used in the FPU compute kernel
   **Key Findings**: DeepWiki was unavailable. Found the answer in `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 206-257): `binary_dest_reuse_tiles` performs a binary operation where one operand comes from DST (moved to SrcA or SrcB) and the other from a CB. Template parameters control the operation type and which source register receives the DST value.

2. **Query**: "What is the difference between FPU and SFPU compute kernel paths?"
   **Reason**: Understanding why two kernel variants exist
   **Key Findings**: DeepWiki was unavailable. Found the answer in `METALIUM_GUIDE.md` (lines 406-444) and `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`: FPU ops work with SrcA/SrcB registers fed from CBs, while SFPU ops work entirely within DST registers. SFPU path is used when FP32 accumulation is needed.

3. **Query**: "How does circular buffer synchronization work?"
   **Reason**: Understanding the producer-consumer protocol between kernels
   **Key Findings**: DeepWiki was unavailable. Found the answer in `METALIUM_GUIDE.md` (lines 78-89, 130-172): `cb_wait_front` blocks until the producer has pushed the specified number of tiles; `cb_pop_front` frees consumed tiles. `tile_regs_acquire/commit` synchronize Math and Pack threads within the compute kernel.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding helper functions used in compute kernels
   **Key Information**: Provides `_with_dt` wrappers (add_tiles_init_with_dt, mul_tiles_init_with_dt, pack_tile_with_dt, etc.) that add data format reconfiguration when FP32_DEST_ACC_EN is defined. Also provides composite helpers like `mul_tiles_to_cb`, `add_tiles_to_cb`, `sub_tiles_to_cb` which handle the full cb_wait/acquire/op/commit/wait/pack/release/pop/push cycle.

2. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used in the program factory
   **Key Information**: `create_cb(cb_index, program, core_spec, page_size, num_pages, data_format)` creates a circular buffer and returns `(cb_index, cb_handle)`. Configures `CircularBufferConfig` with `num_pages * page_size` total size.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Understanding how per-channel scalars are broadcast across tiles
   **Key Information**: `fill_tile_with_first_element_bfloat16(cb_id)` reads element [0] as uint16, packs it into uint32 (duplicated), and writes to all 512 uint32 positions (1024 bfloat16 elements). `fill_tile_with_first_element<float>(cb_id)` reads element [0] as float and writes to all 1024 float positions. `fill_with_val_bfloat16(cb_id, packed_scalar)` fills all 512 uint32 positions with a pre-packed scalar. `fill_with_val<1024, float>(cb_id, scalar)` fills all 1024 float positions.

4. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: Understanding exact signatures and documentation for binary operations
   **Key Information**: Full documentation for `add_tiles`, `sub_tiles`, `mul_tiles`, `binary_dest_reuse_tiles`, `binary_dest_reuse_tiles_init` with parameter descriptions and valid ranges.

5. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Understanding SFPU binary operations used in the FP32 path
   **Key Information**: `add_binary_tile(idst0, idst1, odst)`, `sub_binary_tile(idst0, idst1, odst)`, `mul_binary_tile(idst0, idst1, odst)` all operate on pairs of DST register slots. Maximum 4 tiles per operand (16-bit) or 2 tiles per operand (32-bit) in DST at once.

6. **Source**: `METALIUM_GUIDE.md`
   **Reason**: Understanding overall architecture, register model, and compute kernel structure
   **Key Information**: Three kernel types coordinate via circular buffers. Compute kernel compiles to three cores (Unpack, Math, Pack). `tile_regs_acquire` is for Math core, `tile_regs_wait` is for Pack core -- they synchronize the handoff of DST register ownership. FPU operations read from SrcA/SrcB (fed by unpack from CBs), SFPU operations read from DST directly.

7. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Understanding core distribution API
   **Key Information**: `split_work_to_cores(CoreCoord grid_size, uint32_t units_to_divide, bool row_wise)` returns `(num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2)`. Splits work evenly with remainder distributed across core_group_1 and core_group_2.
