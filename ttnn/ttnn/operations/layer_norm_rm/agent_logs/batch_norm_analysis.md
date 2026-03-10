# Batch Norm Implementation Analysis

## Overview

Batch normalization applies per-channel normalization to a 4D input tensor using pre-computed batch mean and variance statistics. The mathematical operation is:

```
output = (input - batch_mean) / sqrt(batch_var + eps) * weight + bias
```

where `weight` and `bias` are optional affine parameters.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Key architectural insight**: Unlike layer normalization, batch_norm receives pre-computed per-channel statistics (mean, variance) as separate input tensors. The compute kernel does NOT compute mean/variance -- it only applies the normalization formula. This is a critical difference for the layer_norm_rm use case, which must compute statistics inline.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Two-level: outer loop over "channel groups" (complete_iterations + remainder), inner loop over spatial tiles within a channel group |

### Channel-group broadcasting pattern

The compute kernel organizes work around a concept called `tile_freq` (= `cHt * cWt`, i.e., the number of spatial tiles per channel). For each channel, one set of statistics (mean, var, weight, bias) is loaded and reused across all `tile_freq` spatial tiles. This is the fundamental **multi-pass data reuse** pattern.

**How it works**:
- `tile_freq = cHt * cWt` -- spatial tiles per channel
- `tile_start = start_tile_id % tile_freq` -- offset into the first channel group (handles partial channels at core boundaries)
- `complete_iterations = (num_tiles + tile_start) / tile_freq` -- full channel groups to process
- `remaining_iterations = (num_tiles + tile_start) % tile_freq` -- partial trailing channel group

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | Batch Mean | Batch Var | Weight (optional) | Bias (optional) |
|----------|-----------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] or [N, C, 1, 1] | same as mean | same as mean | same as mean |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | matches input | matches input | matches input | matches input |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | same as input [N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | matches input (or optional dtype) |

### Scalar Broadcasting Pattern

The statistics tensors (mean, var, weight, bias) are per-channel scalars stored as tiles where only the first element matters. The writer kernel reads these tiles from DRAM, then calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the scalar across the entire tile in L1. This makes them compatible with element-wise tile operations in the compute kernel. The `FILL_TILE_WITH_FIRST_ELEMENT` macro resolves to:
- `fill_tile_with_first_element<float>` for FLOAT32
- `fill_tile_with_first_element_bfloat16` for BFLOAT16

These functions read element [0] from the CB write pointer and replicate it across all 1024 (float32) or 512 (packed bfloat16) positions in the tile.

## Data Flow Pattern

### High-Level Data Flow

```
DRAM                    Reader              Compute             Writer              DRAM
 |                        |                   |                   |                  |
 |--input tiles---------->| cb_input(c_0)     |                   |                  |
 |                        |===================|>                  |                  |
 |                        | cb_eps(c_4)       |                   |                  |
 |                        |=====[PROGRAM]=====|>                  |                  |
 |                        |                   |                   |                  |
 |--mean tiles------------|-------------------|-->cb_batch_mean(c_1)                 |
 |--var tiles-------------|-------------------|-->cb_batch_var(c_3)                  |
 |--weight tiles----------|-------------------|-->cb_weight(c_5)                     |
 |--bias tiles------------|-------------------|-->cb_bias(c_6)                       |
 |                        |                   |                   |                  |
 |                        |                   | cb_den(c_7)       |                  |
 |                        |                   | cb_tmp_1(c_8)     |                  |
 |                        |                   |                   |                  |
 |                        |                   |==>cb_output(c_2)==|>                 |
 |                        |                   |                   |--output tiles---->|
```

### Stage-by-Stage Flow

| Stage | Kernel | Reads From | Writes To | Operations |
|-------|--------|------------|-----------|------------|
| 1 | Reader | DRAM (input) | cb_input (c_0) | `noc_async_read_tile` per input tile |
| 2 | Reader | runtime arg (eps) | cb_eps (c_4) | `fill_with_val` once at startup |
| 3 | Writer | DRAM (mean, var, weight, bias) | cb_batch_mean (c_1), cb_batch_var (c_3), cb_weight (c_5), cb_bias (c_6) | `noc_async_read_tile` + `FILL_TILE_WITH_FIRST_ELEMENT` per channel |
| 4 | Compute | cb_batch_var (c_3), cb_eps (c_4) | cb_den (c_7) | `add_tiles` + `rsqrt_tile` -- once per channel |
| 5 | Compute | cb_input (c_0), cb_batch_mean (c_1), cb_den (c_7) | cb_tmp_1 or cb_output (c_8/c_2) | `sub_tiles` + `binary_dest_reuse_tiles(ELWMUL)` -- per tile |
| 6 | Compute | cb_tmp_1 (c_8), cb_weight (c_5) | cb_tmp_1 or cb_output (c_8/c_2) | `mul_tiles` -- per tile (optional) |
| 7 | Compute | cb_tmp_1 (c_8), cb_bias (c_6) | cb_output (c_2) | `add_tiles` -- per tile (optional) |
| 8 | Writer | cb_output (c_2) | DRAM (output) | `noc_async_write_tile` per output tile |

**Important naming caveat**: The writer kernel is responsible for reading statistics (mean, var, weight, bias) from DRAM into CBs -- despite being named "writer". This is a common pattern in tt-metal where the writer kernel handles auxiliary inputs that the reader cannot handle due to NoC assignment constraints.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input data tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | cb_batch_mean | Per-channel mean (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group (persists across spatial tiles) |
| c_2 | cb_output | Final output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | cb_batch_var | Per-channel variance (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group (consumed once per channel) |
| c_4 | cb_eps | Epsilon scalar (broadcast-filled) | 2 tiles | 1 tile | Double | Reader | Compute | Program (loaded once, persists entire kernel) |
| c_5 | cb_weight | Per-channel weight (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group (persists across spatial tiles) |
| c_6 | cb_bias | Per-channel bias (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group (persists across spatial tiles) |
| c_7 | cb_den | Intermediate: 1/sqrt(var+eps) | 2 tiles | 1 tile | Double | Compute | Compute | Channel group (computed once, reused across spatial tiles) |
| c_8 | cb_tmp_1 | Intermediate: normalized result before affine | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile, transient) |

### Multi-Pass Data Reuse Pattern (Critical for layer_norm_rm reference)

The key insight is how CBs are managed to enable **reuse of per-channel data across spatial tiles**:

1. **Program-lifetime CB**: `cb_eps` (c_4) is loaded once by the reader at startup with `cb_wait_front(cb_eps, onetile)` at the top of `kernel_main()`, and only popped at the very end with `cb_pop_front(cb_eps, onetile)`. It persists for the entire kernel execution.

2. **Channel-group-lifetime CBs**: `cb_batch_mean` (c_1), `cb_den` (c_7), `cb_weight` (c_5), `cb_bias` (c_6) are loaded/computed once per channel call to `batchnorm_bcast_tiles()` and reused across all spatial tiles (`tile_start` to `freq`). They are popped at the END of `batchnorm_bcast_tiles()`, after the inner spatial loop completes.

3. **Tile-lifetime CBs**: `cb_input` (c_0), `cb_output` (c_2), `cb_tmp_1` (c_8) are produced and consumed per individual tile within the inner loop.

### CB Output Routing Logic

The compute kernel dynamically routes output based on which optional operations are active:

```
cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0
cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0
```

This means:
- **No weight, no bias**: sub+mul result goes directly to `cb_output` (c_2)
- **Weight only**: sub+mul result goes to `cb_tmp_1` (c_8), then weight*result goes to `cb_output` (c_2)
- **Weight + bias**: sub+mul goes to `cb_tmp_1`, weight*result goes to `cb_tmp_1`, bias+result goes to `cb_output`
- **Bias only**: sub+mul goes to `cb_tmp_1`, bias+result goes to `cb_output`

## Pipeline Pattern Summary

All CBs are allocated with capacity = 2 tiles and block size = 1 tile, indicating **double buffering** throughout. This allows the reader/writer and compute to overlap: while compute processes tile N, the reader/writer can fill the CB with tile N+1.

## Index Calculations

### Output tile linearization

Output tiles are linearized in row-major order: `tile_id = n * (C * Ht * Wt) + c * (Ht * Wt) + t` where `t` is the spatial tile index within a channel. The `start_tile_id` for each core is a cumulative sum of tiles assigned to prior cores.

### Channel decomposition from linear tile ID

```
tiles_per_batch = HtWt * C
start_n = start_tile_id / tiles_per_batch
start_remaining = start_tile_id % tiles_per_batch
start_c = start_remaining / HtWt
start_t = start_remaining % HtWt
```

### Reader tile offset calculation

The reader computes a tile offset into the input tensor using stride-based addressing:
```
tile_offset = start_n * n_stride + start_c * c_stride + start_t
```
Where:
- `n_stride = aHt * aWt * aC * (aN > 1)` -- stride between batches (0 if N=1)
- `c_stride = aHt * aWt * (aC > 1)` -- stride between channels (0 if C=1)

The `(aN > 1)` and `(aC > 1)` conditionals prevent stride multiplication when the dimension is 1, allowing for broadcasting.

### Writer statistics offset

The writer uses an analogous stride computation for the statistics tensors (mean, var, weight, bias), using the statistics tensor's own shape dimensions (bN, bC, bHt, bWt). The tile offset indexes into the per-channel tile for each (n, c) pair.

### Compute frequency/counter mechanism

The compute kernel receives:
- `tile_freq = cHt * cWt` -- number of spatial tiles per channel
- `tile_start = start_tile_id % cHtWt` -- starting offset within first channel group

These drive the two-level loop structure where the inner loop processes `tile_start..tile_freq` spatial tiles with shared statistics.

## Memory Access Patterns

### Read Pattern (Reader kernel -- brief)
- **Input tiles**: Sequential within a channel's spatial tiles, then jumps to next channel, then next batch. Uses `noc_async_read_tile` with TensorAccessor for automatic bank mapping.
- **Epsilon**: Written once as a filled tile via `fill_with_val`, no DRAM read.

### Read Pattern (Writer kernel -- brief)
- **Statistics tiles**: One read per channel for each of mean, var, weight, bias. Each is broadcast-filled after reading via `FILL_TILE_WITH_FIRST_ELEMENT`.
- Pattern: for each (n, c), read mean[n,c] and var[n,c] (and optionally weight[c], bias[c]), then write all spatial output tiles.

### Write Pattern (Writer kernel -- brief)
- **Output tiles**: Sequential writes using linearized `start_tile_id + num_tiles_written` as the tile index.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (used as 1D row-major) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores_x * num_cores_y` (all available compute cores) |
| **Work per core** | `num_output_tiles / num_cores` tiles (approximately) |
| **Load balancing** | Two-group split via `split_work_to_cores`: core_group_1 gets `ceil(tiles/cores)` tiles, core_group_2 gets `floor(tiles/cores)` tiles |
| **Remainder handling** | Cores not in either group receive zero-filled args and early-exit (`num_tiles == 0` check in compute kernel) |

All kernels are deployed to `all_device_cores` (the full compute grid), but cores without work skip execution due to the `num_tiles == 0` guard.

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if weight tensor present, 0 otherwise |
| 1 | bias_has_value | uint32_t | 1 if bias tensor present, 0 otherwise |
| 2 | cb_input | uint32_t | CB index for input data (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon constant (c_4) |
| 7 | cb_den | uint32_t | CB index for 1/sqrt(var+eps) intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for weight tensor (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for intermediate scratch (c_8) |
| 10 | cb_bias | uint32_t | CB index for bias tensor (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | tile_freq | uint32_t | Spatial tiles per channel (cHt * cWt) |
| 2 | tile_start | uint32_t | Starting offset within first channel group |

### Reader Compile-Time Arguments (brief)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src | uint32_t | CB index for input (c_0) |
| 1 | cb_id_eps | uint32_t | CB index for epsilon (c_4) |
| 2+ | TensorAccessorArgs | ... | Auto-appended tensor accessor config |

### Reader Runtime Arguments (brief)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value packed as uint32 |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | Starting tile for this core |
| 3 | num_tiles | uint32_t | Tiles to read |
| 4 | HtWt | uint32_t | Spatial tiles per channel |
| 5 | n_stride | uint32_t | Tile stride between batches |
| 6 | c_stride | uint32_t | Tile stride between channels |
| 7 | N | uint32_t | Number of batches |
| 8 | C | uint32_t | Number of channels |

### Writer Compile-Time Arguments (brief)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | Weight present flag |
| 1 | bias_has_value | uint32_t | Bias present flag |
| 2 | cb_id_src (mean) | uint32_t | CB index for batch mean (c_1) |
| 3 | cb_id_dst (output) | uint32_t | CB index for output (c_2) |
| 4 | cb_id_batch_var | uint32_t | CB index for batch var (c_3) |
| 5 | cb_id_weight | uint32_t | CB index for weight (c_5) |
| 6 | cb_id_bias | uint32_t | CB index for bias (c_6) |
| 7+ | TensorAccessorArgs x5 | ... | For mean, output, var, weight, bias buffers |

### Writer Runtime Arguments (brief)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Mean tensor buffer address |
| 1 | batch_var_addr | uint32_t | Variance tensor buffer address |
| 2 | weight_addr | uint32_t | Weight buffer address (0 if absent) |
| 3 | bias_addr | uint32_t | Bias buffer address (0 if absent) |
| 4 | output_addr | uint32_t | Output buffer address |
| 5 | start_tile_id | uint32_t | Starting tile |
| 6 | num_tiles | uint32_t | Tiles to write |
| 7-13 | HtWt, n_stride, c_stride, N, C, Ht, Wt | uint32_t | Shape/stride info |

## Kernel Implementations

### Compute Kernel (FPU path) -- DETAILED ANALYSIS

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`

**Selected when**: `fp32_dest_acc_en == false`

**Includes**:
- `api/compute/eltwise_binary.h` -- provides `add_tiles`, `sub_tiles`, `mul_tiles`, `binary_dest_reuse_tiles`
- `ttnn/kernel/compute/moreh_common.hpp` -- provides `pack_tile_with_dt`, `add_tiles_init_with_dt`, `sub_tiles_init_with_dt`, `mul_tiles_init_with_dt`

**Initialization**: `binary_op_init_common(cb_other, cb_bcast, cb_output_0)` -- configures unpack, math, and pack units for binary operations.

#### Function: `batchnorm_bcast_tiles` (FPU path)

Full signature:
```cpp
ALWI void batchnorm_bcast_tiles(
    uint32_t cb_bcast,      // cb_batch_mean -- per-channel mean
    uint32_t cb_other,      // cb_input -- input data tiles
    uint32_t freq,          // spatial tiles to process (up to HtWt)
    uint32_t tile_start,    // starting spatial tile offset
    uint32_t cb_batch_var,  // per-channel variance
    uint32_t cb_eps,        // epsilon constant tile
    uint32_t cb_den,        // intermediate: 1/sqrt(var+eps)
    uint32_t cb_weight,     // optional per-channel weight
    uint32_t cb_bias,       // optional per-channel bias
    uint32_t cb_tmp_1,      // scratchpad for intermediate results
    uint32_t cb_output_0,   // final output CB
    uint32_t weight_has,    // weight present flag
    uint32_t bias_has)      // bias present flag
```

#### Phase 1: Compute denominator -- rsqrt(var + eps)

This phase runs ONCE per channel (outside the spatial tile loop).

```cpp
cb_reserve_back(cb_den, onetile);
cb_wait_front(cb_batch_var, onetile);

tile_regs_acquire();
add_tiles_init_with_dt(cb_batch_var, cb_eps);   // configure unpacker for var+eps data formats
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);    // dst0 = var + eps
rsqrt_tile_init();
rsqrt_tile(dst0);                                // dst0 = 1/sqrt(var + eps)
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(dst0, cb_den);                 // pack to cb_den
tile_regs_release();

cb_pop_front(cb_batch_var, onetile);             // consume var tile
cb_push_back(cb_den, onetile);                   // publish denominator
```

**CB state after Phase 1**: `cb_den` has 1 tile available. `cb_batch_var` consumed. `cb_eps` remains (not popped -- program lifetime).

#### Phase 2: Wait for per-channel data

```cpp
cb_wait_front(cb_bcast, onetile);    // wait for mean tile
cb_wait_front(cb_den, onetile);      // wait for denominator (just computed)
if (weight_has_value) cb_wait_front(cb_weight, onetile);
if (bias_has_value) cb_wait_front(cb_bias, onetile);
```

These CBs are now "pinned" for the duration of the inner spatial loop.

#### Phase 3: Per-tile normalization (inner loop: `tile_start` to `freq`)

For each spatial tile:

**Step 3a: Subtract mean (using `binary_dest_reuse_tiles` pattern)**
```cpp
cb_wait_front(cb_other, onetile);           // wait for input tile
cb_reserve_back(cb_affine_or_out, onetile); // reserve output slot

tile_regs_acquire();
sub_tiles_init(cb_other, cb_bcast);
sub_tiles(cb_other, cb_bcast, 0, 0, 0);    // dst0 = input - mean
```

**Step 3b: Multiply by denominator (dest reuse -- NO intermediate CB write)**
```cpp
binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
// dst0 = (input - mean) * 1/sqrt(var+eps)
// DEST_TO_SRCA: dst0 value moves to SrcA, cb_den[0] unpacks to SrcB, result = SrcA * SrcB -> dst0
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(0, cb_affine_or_out);     // pack normalized result
tile_regs_release();

cb_push_back(cb_affine_or_out, onetile);
cb_pop_front(cb_other, onetile);             // consume input tile
```

**Key optimization**: `binary_dest_reuse_tiles` avoids writing the `(input - mean)` intermediate to a CB and re-reading it. The subtraction result stays in the DST register and is directly used as SrcA for the multiplication. This saves one CB write + one CB read per tile.

**Step 3c: Optional weight multiplication**
```cpp
if (weight_has_value) {
    cb_reserve_back(cb_scaled_output, onetile);
    cb_wait_front(cb_affine_or_out, 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
    mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_scaled_output);
    tile_regs_release();

    cb_pop_front(cb_affine_or_out, 1);
    cb_push_back(cb_scaled_output, onetile);
}
```

**Step 3d: Optional bias addition**
```cpp
if (bias_has_value) {
    cb_reserve_back(cb_output_0, onetile);
    cb_wait_front(cb_tmp_1, onetile);

    tile_regs_acquire();
    add_tiles_init_with_dt(cb_tmp_1, cb_bias);
    add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_output_0);
    tile_regs_release();

    cb_pop_front(cb_tmp_1, onetile);
    cb_push_back(cb_output_0, onetile);
}
```

#### Phase 4: Release per-channel data

```cpp
cb_pop_front(cb_bcast, onetile);    // release mean
cb_pop_front(cb_den, onetile);      // release denominator
if (weight_has_value) cb_pop_front(cb_weight, onetile);
if (bias_has_value) cb_pop_front(cb_bias, onetile);
```

#### kernel_main() structure

```cpp
void kernel_main() {
    // Runtime args
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    // Compile-time args (constexpr)
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;
    // CB indices from compile-time args 2-10

    binary_op_init_common(cb_other, cb_bcast, cb_output_0);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    cb_wait_front(cb_eps, onetile);  // pin eps for entire kernel

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(...);  // process one full channel group
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(...);  // process partial trailing group
    }

    cb_pop_front(cb_eps, onetile);   // release eps
}
```

Note: `tile_start` is reset to 0 after the first iteration (`tile_start = 0` in the for-loop increment), because only the first channel group can be partial (due to core boundary splitting).

### Compute Kernel (SFPU path) -- DIFFERENCES FROM FPU

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`

**Selected when**: `fp32_dest_acc_en == true`

**Key differences**:
1. **Initialization**: Uses `unary_op_init_common(cb_other, cb_output_0)` instead of `binary_op_init_common`
2. **Includes**: `api/compute/eltwise_binary_sfpu.h`, `api/compute/eltwise_unary/sfpu_split_includes.h`, `api/compute/eltwise_unary/eltwise_unary.h`, `api/compute/eltwise_unary/rsqrt.h`
3. **Binary operations use explicit copy_tile + SFPU binary ops**:
   - Instead of `add_tiles(cb_batch_var, cb_eps, ...)`, the SFPU path does:
     ```cpp
     copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
     copy_tile(cb_batch_var, 0, 0);   // dst[0] = batch_var
     add_binary_tile_init();
     copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
     copy_tile(cb_eps, 0, 1);         // dst[1] = eps
     add_binary_tile(0, 1, 0);        // dst[0] = dst[0] + dst[1]
     ```
   - Similarly for subtraction: `sub_binary_tile(i*2, i*2+1, i*2)`
   - And multiplication: `mul_binary_tile(i*2, i*2+1, i*2)`
4. **Pack function**: Uses `pack_tile(i*2, cb_den)` directly instead of `pack_tile_with_dt`
5. **Tile register management**: `tile_regs_acquire()` and `tile_regs_wait()` are called back-to-back before operations (different synchronization pattern from FPU)
6. **DST register addressing**: Uses `i*2` and `i*2+1` for two operands in DST registers, whereas FPU path uses tile indices 0,0 in CB

### Kernel Specification Table

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM (input tensor) | c_0 (input), c_4 (eps) | Read input tiles; fill eps tile once |
| batch_norm_kernel / batch_norm_sfpu_kernel | RISCV_2 | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2 (output), c_7 (den), c_8 (tmp) | sub, mul (dest_reuse), rsqrt, optional mul/add |
| writer_batch_norm | RISCV_1 | NOC1 | DRAM (mean, var, weight, bias) + c_2 | c_1, c_3, c_5, c_6 + DRAM (output) | Read statistics, broadcast-fill, write output tiles |

## Implementation Notes

### Two compute kernel variants

The program factory selects between `batch_norm_kernel.cpp` (FPU path) and `batch_norm_sfpu_kernel.cpp` (SFPU path) based on `fp32_dest_acc_en`:

```cpp
fmt::format("...batch_norm_{}.cpp", fp32_dest_acc_en ? "sfpu_kernel" : "kernel")
```

When `fp32_dest_acc_en` is true, all CBs listed in the `unpack_to_dest_mode` array are set to `UnpackToDestMode::UnpackToDestFp32`, ensuring full FP32 precision in the DST accumulator registers.

### UnpackToDestMode configuration

```cpp
std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
if (fp32_dest_acc_en) {
    for (const auto cb_index : {input, mean, var, eps, den, weight, tmp_1, bias}) {
        unpack_to_dest_mode[cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }
}
```

This is critical: it tells the unpacker to convert data to FP32 when loading into DST registers, enabling full-precision computation.

### Data format defines for reader/writer

The program factory sets preprocessor defines for scalar filling functions based on input data type:
- FLOAT32: `FILL_TILE_WITH_FIRST_ELEMENT = fill_tile_with_first_element<float>`, `FILL_WITH_VALUE_FLOAT = fill_with_val<1024, float>`
- BFLOAT16: `FILL_TILE_WITH_FIRST_ELEMENT = fill_tile_with_first_element_bfloat16`, `FILL_WITH_VALUE = fill_with_val_bfloat16`

Note the asymmetry: FLOAT32 uses `FILL_WITH_VALUE_FLOAT` while BFLOAT16 uses `FILL_WITH_VALUE` (different define names).

### Relevance to layer_norm_rm

For implementing layer_norm_rm, the key patterns to extract from batch_norm are:

1. **The `binary_dest_reuse_tiles` pattern**: Eliminates intermediate CB writes when chaining operations (sub then mul). This is directly applicable to layer_norm_rm for `(x - mean) * rsqrt(var + eps)`.

2. **Scalar CB setup**: The `fill_with_val` / `FILL_TILE_WITH_FIRST_ELEMENT` patterns for filling tiles with constants. Layer_norm_rm will need this for epsilon.

3. **CB lifetime management**: The three-tier lifetime (program, channel-group, per-tile) is a powerful pattern. For layer_norm_rm, the analogous structure would be:
   - Program lifetime: epsilon constant
   - Row-group lifetime: per-row mean and variance (computed, not pre-supplied)
   - Per-tile lifetime: input tiles and output tiles

4. **The `rsqrt_tile` SFPU operation**: Used identically in both batch_norm and what layer_norm_rm needs.

5. **Dynamic CB routing via `cb_affine_or_out` / `cb_scaled_output`**: Shows how to handle optional gamma/beta without separate kernel variants.

6. **Two kernel variants (FPU vs SFPU)**: The pattern of selecting between FPU and SFPU compute paths based on `fp32_dest_acc_en`.

**Critical difference**: batch_norm receives pre-computed statistics, while layer_norm_rm must compute row-wise mean and variance inline using reduction operations. This means layer_norm_rm will need additional compute phases (tilize, reduce_sum for mean, subtract+square+reduce for variance) before the normalization phase that resembles batch_norm's compute logic.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do the compute kernel APIs work in tt-metal? Specifically: FPU vs SFPU path, binary_dest_reuse_tiles, tile_regs lifecycle, pack_tile_with_dt"
   **Reason**: Needed to understand the two compute kernel variants and the dest-reuse optimization pattern.
   **Key Findings**: FPU path uses `add_tiles`/`sub_tiles`/`mul_tiles` which handle SrcA/SrcB register management automatically. SFPU path requires explicit `copy_tile` into DST registers then binary tile ops. `binary_dest_reuse_tiles` with `DEST_TO_SRCA` keeps the previous result in DST, moves it to SrcA, unpacks CB tile to SrcB, and performs the operation -- eliminating intermediate CB roundtrips. `pack_tile_with_dt` calls `pack_reconfig_data_format` before `pack_tile` when `FP32_DEST_ACC_EN` is defined.

2. **Query**: "What does FILL_TILE_WITH_FIRST_ELEMENT do in tt-metal dataflow kernels?"
   **Reason**: Needed to understand how per-channel scalars are broadcast across tiles.
   **Key Findings**: Reads element [0] from the CB tile and writes it to all positions (512 uint32 writes for bfloat16, 1024 for float32). Used after reading a per-channel statistic tile from DRAM to make it uniform for element-wise operations.

3. **Query**: "binary_dest_reuse_tiles function signatures and DEST_TO_SRCA mechanism"
   **Reason**: Critical optimization pattern used in the compute kernel for chaining sub+mul without intermediate CB write.
   **Key Findings**: `binary_dest_reuse_tiles_init<EltwiseBinaryType, EltwiseBinaryReuseDestType>(uint32_t icb0)` and `binary_dest_reuse_tiles<...>(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)`. DEST_TO_SRCA loads DST[dst_tile_index] into SrcA, unpacks CB tile into SrcB, performs operation, writes result back to DST[dst_tile_index].

4. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: Understanding core distribution and load balancing for multi-core execution.
   **Key Findings**: Returns (num_cores, all_cores, core_group_1, core_group_2, tiles_per_group_1, tiles_per_group_2). Group 1 gets `ceil(tiles/cores)` tiles, group 2 gets `floor(tiles/cores)`. Group 1 has `tiles % cores` members.

5. **Query**: "Difference between binary_op_init_common and unary_op_init_common"
   **Reason**: Understanding why the two compute variants use different initialization.
   **Key Findings**: `binary_op_init_common` configures unpack for two input CBs (AB mode), math for binary ops, and pack for output. `unary_op_init_common` configures for single input CB. SFPU kernel uses unary init because SFPU binary ops manage their own operand setup through explicit `copy_tile` calls.

6. **Query**: "SFPU vs FPU binary operation APIs and when to use each"
   **Reason**: Understanding the fp32_dest_acc_en selection logic.
   **Key Findings**: FPU path is simpler (automatic register management) and used for standard precision. SFPU path provides explicit DST register control, needed when `fp32_dest_acc_en=true` to maintain FP32 precision throughout the compute pipeline. SFPU binary ops operate directly on DST registers, enabling operation chaining without CB intermediate writes.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Header included by both compute kernels, provides `_with_dt` wrapper functions.
   **Key Information**: `pack_tile_with_dt` calls `pack_reconfig_data_format(icb)` before `pack_tile` when `FP32_DEST_ACC_EN` is defined. Similarly, `add_tiles_init_with_dt`, `sub_tiles_init_with_dt`, `mul_tiles_init_with_dt` call `reconfig_data_format(icb0, icb1)` before their base init functions. Also provides `copy_tile_to_dst`, `pack_tile_from_dst`, and various composite helpers like `mul_tiles_to_cb`, `add_tiles_to_cb`.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Provides tile-filling utility functions used by reader and writer.
   **Key Information**: Contains `fill_with_val_bfloat16(cb_id, packed_scalar)`, `fill_with_val<N, T>(cb_id, scalar)`, `fill_tile_with_first_element_bfloat16(cb_id)`, `fill_tile_with_first_element<T>(cb_id)`, plus row/column broadcast variants.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` helper used by the program factory.
   **Key Information**: `create_cb(cb_index, program, core_spec, page_size, num_pages, data_format)` creates a CircularBufferConfig with `num_pages * page_size` total size and returns `(cb_index, handle)`.

4. **Source**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.hpp`
   **Reason**: Understanding the operation's interface, tensor args, and attributes.
   **Key Information**: `tensor_args_t` has 6 fields: input, batch_mean, batch_var, optional weight, optional bias, optional output. `operation_attributes_t` has eps, memory_config, compute_kernel_config, input_dtype, optional dtype.
