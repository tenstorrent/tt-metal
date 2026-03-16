# Batch Norm Implementation Analysis (Compute-Core Reference)

## Overview

**Batch normalization** normalizes each element of an input tensor by subtracting the batch mean and dividing by the batch standard deviation (computed from batch variance), with optional affine transformation (gamma/beta). The math is:

```
output = (input - batch_mean) / sqrt(batch_var + eps) * weight + bias
```

Unlike layer norm, batch norm receives pre-computed mean and variance as separate input tensors (one per channel), rather than computing them inline. This analysis focuses on the **compute kernel structure** as a reference for implementing layer_norm_rm, which must additionally compute mean and variance from input data.

**Program Factory**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Two Compute Kernel Variants**:
- `batch_norm_kernel.cpp` -- FPU-path (uses `eltwise_binary.h` APIs that operate directly on CB tiles)
- `batch_norm_sfpu_kernel.cpp` -- SFPU-path (uses `eltwise_binary_sfpu.h` APIs with explicit `copy_tile` into DST registers)

The variant is selected at kernel creation time based on `fp32_dest_acc_en`:
```cpp
fmt::format("...batch_norm_{}.cpp", fp32_dest_acc_en ? "sfpu_kernel" : "kernel")
```

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Per-channel iteration: within each channel (HtWt spatial tiles), the same mean/var/weight/bias tile is broadcast across all spatial tiles |

---

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | batch_mean | batch_var | weight (gamma) | bias (beta) |
|----------|-----------|------------|-----------|----------------|-------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] or [N, C, 1, 1] | same as batch_mean | same as batch_mean | same as batch_mean |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 / FLOAT32 | any numeric | any numeric | any numeric | any numeric |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | configurable via `operation_attributes.dtype` |

### Layout Transformations

None. All tensors are already in TILE_LAYOUT. The batch_mean/var/weight/bias tensors have their first element broadcast to fill the entire tile via `FILL_TILE_WITH_FIRST_ELEMENT` in the writer kernel (since these are per-channel scalars stored as tiles).

---

## Data Flow Pattern

### High-Level Flow

```
Reader:  input[x] --> CB_c0 (input tiles)
         eps      --> CB_c4 (epsilon scalar tile, filled once at start)

Writer:  batch_mean  --> CB_c1 (one per channel, broadcast-filled)
         batch_var   --> CB_c3 (one per channel, broadcast-filled)
         weight      --> CB_c5 (one per channel, broadcast-filled, optional)
         bias        --> CB_c6 (one per channel, broadcast-filled, optional)
         CB_c2       --> output DRAM (write computed results)

Compute: CB_c0 (input) + CB_c1 (mean) + CB_c3 (var) + CB_c4 (eps)
         + CB_c5 (weight) + CB_c6 (bias)
         --> CB_c7 (intermediate: inv_std)
         --> CB_c8 (intermediate: temp_1)
         --> CB_c2 (output)
```

### Important Design Note: Writer Does Reading

In this operation, the **writer kernel** is responsible for reading batch_mean, batch_var, weight, and bias from DRAM into their respective CBs, in addition to writing the output. The **reader kernel** reads only the input tensor and fills the epsilon CB. This is a common pattern in tt-metal where the naming reflects core assignment (NOC0 vs NOC1), not the direction of data flow.

### Per-Channel Broadcast Pattern

The operation iterates over channels. For each channel:
1. Writer reads one tile of batch_mean, batch_var, (optionally weight, bias) and broadcasts the first element to fill the tile
2. Reader streams spatial tiles (HtWt tiles) for that channel
3. Compute processes each spatial tile against the broadcast channel values
4. Writer drains output tiles

Key: `freq = cHt * cWt` -- this is the number of spatial tiles per channel. The channel parameters (mean, var, weight, bias) persist in their CBs across all spatial tiles within one channel, then get replaced for the next channel.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | input_tensor_cb | Input spatial tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | batch_mean_tensor_cb | Mean per channel (broadcast) | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel: `freq` tiles) |
| c_2 | output_tensor_cb | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | batch_var_tensor_cb | Variance per channel (broadcast) | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel) |
| c_4 | eps_cb | Epsilon scalar tile | 2 tiles | 1 tile | Double | Reader | Compute | Program (filled once, persists) |
| c_5 | weight_tensor_cb | Weight/gamma per channel | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel) |
| c_6 | bias_tensor_cb | Bias/beta per channel | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel) |
| c_7 | den_cb | `1/sqrt(var + eps)` intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Row (per channel, reused for all spatial tiles) |
| c_8 | temp_1_cb | Normalized intermediate (before affine) | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile) |

### Pipeline Pattern Summary

All CBs have capacity=2, block_size=1, giving **double-buffered** configuration throughout. This allows overlap between producer and consumer operations.

### CB Lifetime and Multi-Pass Reuse Pattern (Critical for layer_norm_rm reference)

This is the most important pattern for understanding how batch_norm manages per-channel data:

1. **Program-lifetime CB**: `eps_cb` (c_4) -- filled once by reader at kernel start, consumed by compute at the beginning of each channel iteration. The compute kernel does a single `wait_front` at start and `pop_front` at end of all processing.

2. **Channel-lifetime CBs**: `batch_mean` (c_1), `batch_var` (c_3), `weight` (c_5), `bias` (c_6), `den` (c_7) -- these are loaded once per channel and persist across all spatial tiles in that channel. Inside `batchnorm_bcast_tiles`:
   - `wait_front` is called **before** the spatial tile loop
   - The inner loop reads from these CBs without popping (they remain in the CB)
   - `pop_front` is called **after** the spatial tile loop completes
   - This is the "broadcast" pattern: one value reused for many tiles

3. **Per-tile CBs**: `input` (c_0), `output` (c_2), `temp_1` (c_8) -- consumed and produced per-tile within the inner loop.

**Relevance for layer_norm_rm**: The layer_norm operation will need a similar pattern for:
- The mean tile (computed once per row, reused for all Wt tiles in that row during centering)
- The inv_std tile (computed once per row, reused for all Wt tiles during normalization)
- The gamma/beta tiles (may differ per position within the row)

---

## Compute Kernel Structure -- Deep Dive (FPU-path: `batch_norm_kernel.cpp`)

### Includes and Dependencies

```cpp
#include "api/compute/eltwise_binary.h"      // FPU binary ops: add_tiles, sub_tiles, mul_tiles
#include "ttnn/kernel/compute/moreh_common.hpp" // Helpers: pack_tile_with_dt, *_init_with_dt wrappers
#include "experimental/circular_buffer.h"      // CB object-oriented API
```

### Initialization

```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```

This initializes the three compute threads (UNPACK, MATH, PACK) for binary operations:
- Configures the unpacker for `cb_other` (input) and `cb_bcast` (mean)
- Sets up math unit for binary operations
- Configures packer for `cb_output_0`

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | Whether gamma tensor is present (0 or 1) |
| 1 | bias_has_value | uint32_t | Whether beta tensor is present (0 or 1) |
| 2 | cb_input | uint32_t | CB index for input tiles (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for mean tiles (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output tiles (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for variance tiles (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon tile (c_4) |
| 7 | cb_den | uint32_t | CB index for inv_std intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for weight/gamma tiles (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for temp intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB index for bias/beta tiles (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total output tiles assigned to this core |
| 1 | tile_freq | uint32_t | Spatial tiles per channel (HtWt = cHt * cWt) |
| 2 | tile_start | uint32_t | Starting offset within first channel (`start_tile_id % cHtWt`) |

### Iteration Structure

The compute kernel splits work into complete and partial channel iterations:

```cpp
uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
```

This handles the case where a core's tile assignment doesn't align with channel boundaries. The first iteration may start mid-channel (at `tile_start`), and the last may end mid-channel (at `remaining_iterations`).

### Core Computation: `batchnorm_bcast_tiles` (FPU-path)

This is the main computation function, called once per channel (or partial channel).

**Parameters**: `cb_bcast` (mean CB), `cb_other` (input CB), `freq` (tiles in this channel pass), `tile_start`, plus all intermediate/parameter CB IDs.

**CB aliasing for output routing**:
```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```
This routing ensures the last operation in the chain writes to `cb_output_0` regardless of which optional operations are enabled:
- No weight, no bias: normalize writes directly to `cb_output_0`
- Weight only: normalize writes to `cb_tmp_1`, weight multiply writes to `cb_output_0`
- Weight + bias: normalize writes to `cb_tmp_1`, weight multiply writes to `cb_tmp_1`, bias add writes to `cb_output_0`
- Bias only: normalize writes to `cb_tmp_1`, bias add writes to `cb_output_0`

**Phase 1: Compute inv_std = 1/sqrt(var + eps)** (once per channel)

```cpp
// Wait for variance and epsilon
cb_den_obj.reserve_back(onetile);
cb_batch_var_obj.wait_front(onetile);

tile_regs_acquire();
add_tiles_init_with_dt(cb_batch_var, cb_eps);     // Init: var + eps
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);      // dst0 = var + eps
rsqrt_tile_init();                                  // Init SFPU for rsqrt
rsqrt_tile(dst0);                                   // dst0 = 1/sqrt(var + eps)
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(dst0, cb_den);                   // Pack to den_cb
tile_regs_release();

cb_batch_var_obj.pop_front(onetile);               // Done with variance
cb_den_obj.push_back(onetile);                     // inv_std ready
```

**Key API calls and signatures**:
- `add_tiles_init_with_dt(icb0, icb1)` -- from moreh_common.hpp, wraps `reconfig_data_format(icb0, icb1)` (if FP32_DEST_ACC_EN) + `add_tiles_init(icb0, icb1)`. Initializes the FPU for elementwise addition.
- `add_tiles(icb0, icb1, itile0, itile1, idst)` -- performs `DST[idst] = CB[icb0][itile0] + CB[icb1][itile1]`
- `rsqrt_tile_init()` -- initializes SFPU for reciprocal square root
- `rsqrt_tile(dst_index)` -- computes `DST[dst_index] = 1/sqrt(DST[dst_index])` in-place
- `pack_tile_with_dt(dst_index, cb_id)` -- from moreh_common.hpp, wraps `pack_reconfig_data_format(cb_id)` (if FP32_DEST_ACC_EN) + `pack_tile(dst_index, cb_id)`. Packs DST register to CB.

**Phase 2: Wait for broadcast values** (once per channel)

```cpp
cb_bcast_obj.wait_front(onetile);      // mean ready
cb_den_obj.wait_front(onetile);        // inv_std ready
if (weight_has_value) cb_weight_obj.wait_front(onetile);  // gamma ready
if (bias_has_value)   cb_bias_obj.wait_front(onetile);    // beta ready
```

**Phase 3: Per-tile normalization loop** (iterated `freq - tile_start` times for first channel, `freq` for subsequent)

```cpp
for (uint32_t j = tile_start; j < freq; ++j) {
```

**Step 3a: centered = input - mean**
```cpp
cb_other_obj.wait_front(onetile);
cb_affine_or_out_obj.reserve_back(onetile);

tile_regs_acquire();
sub_tiles_init(cb_other, cb_bcast);              // Init subtraction
sub_tiles(cb_other, cb_bcast, 0, 0, 0);         // dst0 = input - mean
```

**Key API calls**:
- `sub_tiles_init(icb0, icb1)` -- initializes FPU for subtraction (`icb0 - icb1`)
- `sub_tiles(icb0, icb1, itile0, itile1, idst)` -- `DST[idst] = CB[icb0][itile0] - CB[icb1][itile1]`

**Step 3b: normalized = centered * inv_std** (chained in same DST register)
```cpp
binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
tile_regs_commit();
```

**Key API calls**:
- `binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(icb0)` -- Initializes a binary operation that reuses DST as SRCA. The other operand comes from `icb0`.
- `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(icb0, itile, idst)` -- Performs `DST[idst] = DST[idst] * CB[icb0][itile]`. The current DST value (centered) is moved to SRCA, and `cb_den` tile provides SRCB. This avoids an intermediate pack/unpack round-trip.

```cpp
tile_regs_wait();
pack_tile_with_dt(0, cb_affine_or_out);         // Pack normalized result
tile_regs_release();

cb_affine_or_out_obj.push_back(onetile);
cb_other_obj.pop_front(onetile);                 // Done with input tile
```

**Step 3c: scaled = normalized * weight** (optional, if gamma present)
```cpp
if (weight_has_value) {
    cb_scaled_output_obj.reserve_back(onetile);
    cb_affine_or_out_obj.wait_front(1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
    mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);  // dst0 = normalized * gamma
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_scaled_output);
    tile_regs_release();

    cb_affine_or_out_obj.pop_front(1);
    cb_scaled_output_obj.push_back(onetile);
}
```

**Key API calls**:
- `mul_tiles_init_with_dt(icb0, icb1)` -- from moreh_common.hpp, wraps data format reconfig + `mul_tiles_init(icb0, icb1)`
- `mul_tiles(icb0, icb1, itile0, itile1, idst)` -- `DST[idst] = CB[icb0][itile0] * CB[icb1][itile1]`

**Step 3d: output = scaled + bias** (optional, if beta present)
```cpp
if (bias_has_value) {
    cb_output_0_obj.reserve_back(onetile);
    cb_tmp_1_obj.wait_front(onetile);

    tile_regs_acquire();
    add_tiles_init_with_dt(cb_tmp_1, cb_bias);
    add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);  // dst0 = scaled + beta
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_output_0);
    tile_regs_release();

    cb_tmp_1_obj.pop_front(onetile);
    cb_output_0_obj.push_back(onetile);
}
```

**Phase 4: Release broadcast values** (once per channel, after all spatial tiles)

```cpp
cb_bcast_obj.pop_front(onetile);        // Release mean
cb_den_obj.pop_front(onetile);          // Release inv_std
if (weight_has_value) cb_weight_obj.pop_front(onetile);
if (bias_has_value)   cb_bias_obj.pop_front(onetile);
```

### Register Pipeline Pattern

Every compute step follows the same 4-function pattern:

```
tile_regs_acquire()  -->  [compute ops]  -->  tile_regs_commit()
tile_regs_wait()     -->  pack_tile()    -->  tile_regs_release()
```

- `tile_regs_acquire()`: Math core claims DST registers
- `tile_regs_commit()`: Math core releases DST to Packer
- `tile_regs_wait()`: Packer waits until DST is committed
- `tile_regs_release()`: Packer frees DST for next acquire

The one exception is the binary_dest_reuse pattern where `sub_tiles` and `binary_dest_reuse_tiles` are chained within the same acquire/commit block, allowing the DST result of subtraction to be immediately used as an input to multiplication without packing.

---

## SFPU-Path Kernel Differences (`batch_norm_sfpu_kernel.cpp`)

The SFPU-path kernel is selected when `fp32_dest_acc_en = true`. Key differences:

### Initialization
```cpp
unary_op_init_common(cb_other, cb_output_0);  // Instead of binary_op_init_common
```

### All Binary Operations Use Explicit DST Copy Pattern

Instead of the FPU `add_tiles`/`sub_tiles`/`mul_tiles` that operate directly on CB tiles, the SFPU path:
1. Explicitly copies tiles into DST registers using `copy_tile()`
2. Performs SFPU binary operations on DST register pairs
3. Packs results back

**Example: var + eps in SFPU-path**:
```cpp
tile_regs_acquire();
tile_regs_wait();

// Copy var to dst[0]
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
copy_tile(cb_batch_var, 0, 0);    // dst[0] = var

// Copy eps to dst[1] and add
add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
copy_tile(cb_eps, 0, 1);          // dst[1] = eps
add_binary_tile(0, 1, 0);         // dst[0] = dst[0] + dst[1]

// rsqrt
rsqrt_tile_init();
rsqrt_tile(0);                    // dst[0] = 1/sqrt(dst[0])

pack_tile(0, cb_den);             // Pack result
tile_regs_commit();
tile_regs_release();
```

**SFPU binary operation signatures**:
- `copy_tile_to_dst_init_short_with_dt(prev_cb, curr_cb)` -- Reconfigures data format from `prev_cb` to `curr_cb` and initializes copy
- `copy_tile(cb_id, tile_index, dst_index)` -- Copies CB tile to DST register
- `add_binary_tile_init()` -- Initializes SFPU addition
- `add_binary_tile(dst_a, dst_b, dst_result)` -- `DST[dst_result] = DST[dst_a] + DST[dst_b]`
- `sub_binary_tile_init()` -- Initializes SFPU subtraction
- `sub_binary_tile(dst_a, dst_b, dst_result)` -- `DST[dst_result] = DST[dst_a] - DST[dst_b]`
- `mul_binary_tile_init()` -- Initializes SFPU multiplication
- `mul_binary_tile(dst_a, dst_b, dst_result)` -- `DST[dst_result] = DST[dst_a] * DST[dst_b]`

**Key difference from FPU path**: SFPU ops use pairs of DST registers (e.g., `i*2` and `i*2+1`) for operands, while FPU ops read directly from CBs into the unpacker. The SFPU path requires more explicit register management but supports full FP32 accumulation.

### UnpackToDestMode Configuration

When `fp32_dest_acc_en` is true, all input CBs are configured with `UnpackToDestMode::UnpackToDestFp32`:
```cpp
unpack_to_dest_mode[cb_index] = UnpackToDestMode::UnpackToDestFp32;
```
This is set for CBs: input, batch_mean, batch_var, eps, den, weight, temp_1, bias. This ensures tiles are unpacked to FP32 format in the DST registers for maximum precision.

---

## Scalar/Constant CB Setup

### Epsilon Tile (CB c_4)

The epsilon scalar is packed into a full tile by the reader kernel:

**Host-side packing** (program factory):
```cpp
const auto packed_scalar_eps = input_tensor.dtype() == DataType::FLOAT32
    ? std::bit_cast<uint32_t>(scalar)              // FP32: raw bit cast
    : pack_two_bfloat16_into_uint32({scalar, scalar});  // BF16: two values in one u32
```

**Kernel-side filling** (reader):
```cpp
cb_id_eps_obj.reserve_back(onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    float eps_f = 0;
    std::memcpy(&eps_f, &eps, sizeof(float));
    FILL_WITH_VALUE_FLOAT(cb_id_eps, eps_f);      // fill_with_val<1024, float>
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_eps, eps);               // fill_with_val_bfloat16
#endif
cb_id_eps_obj.push_back(onetile);
```

The `fill_with_val` functions write the scalar to every element (1024 elements for FP32, 512 packed u32 words for BF16) of the tile in L1 memory. This creates a tile where every element is the epsilon value.

### Channel Parameter Tiles (mean, var, weight, bias)

Each is a per-channel scalar stored as a tile. The writer reads the tile from DRAM, then calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the first element to all 1024 positions in the tile:

```cpp
noc.async_read(src, cb_id_src_obj, src_tile_bytes, {.page_id = tile_offset}, {.offset_bytes = 0});
noc.async_read_barrier();
FILL_TILE_WITH_FIRST_ELEMENT(cb_id_src);  // Broadcasts [0][0] to all positions
cb_id_src_obj.push_back(onetile);
```

This fills the entire tile with a single scalar value, enabling element-wise operations (add/sub/mul) to function as broadcast operations.

---

## Reader/Writer Responsibilities (High-Level Summary)

### Reader Kernel
- **Provides**: Input spatial tiles (CB c_0), epsilon constant tile (CB c_4)
- **Pattern**: Fills epsilon once at start, then streams input tiles one-at-a-time in N,C,H,W order
- **Key feature**: Tracks strides for N and C dimensions to navigate tiled tensor layout

### Writer Kernel
- **Provides**: batch_mean (CB c_1), batch_var (CB c_3), weight (CB c_5), bias (CB c_6)
- **Consumes**: Output tiles (CB c_2)
- **Pattern**: For each channel, reads one tile each of mean/var/weight/bias, broadcasts to fill, then drains HtWt output tiles for that channel
- **Key feature**: Uses `FILL_TILE_WITH_FIRST_ELEMENT` to convert per-channel scalars to full broadcast tiles

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (but linearized row-major) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | Up to `num_cores_x * num_cores_y` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` |
| **Load balancing** | Two-group split via `split_work_to_cores()` |

The `split_work_to_cores` utility divides total output tiles across cores. Cores in group_1 get one more tile than cores in group_2 (or groups may be equal). Cores not in either group are assigned 0 tiles and return immediately.

---

## Key moreh_common.hpp Helpers (Complete Reference)

These are wrappers from `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` that handle FP32 data format reconfiguration:

### Data Format Wrappers (`_with_dt` variants)

All follow the same pattern -- if `FP32_DEST_ACC_EN` is defined, they call `reconfig_data_format()` before the actual init:

| Helper | Wraps | Signature |
|--------|-------|-----------|
| `pack_tile_with_dt(dst, cb)` | `pack_reconfig_data_format(cb)` + `pack_tile(dst, cb)` | `(uint32_t ifrom_dst, uint32_t icb)` |
| `copy_tile_init_with_dt(cb, transpose)` | `reconfig_data_format_srca(cb)` + `copy_tile_to_dst_init_short(cb, transpose)` | `(uint32_t icb, uint32_t transpose = 0)` |
| `add_tiles_init_with_dt(icb0, icb1)` | `reconfig_data_format(icb0, icb1)` + `add_tiles_init(icb0, icb1)` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` |
| `sub_tiles_init_with_dt(icb0, icb1)` | `reconfig_data_format(icb0, icb1)` + `sub_tiles_init(icb0, icb1)` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` |
| `mul_tiles_init_with_dt(icb0, icb1)` | `reconfig_data_format(icb0, icb1)` + `mul_tiles_init(icb0, icb1)` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` |
| `mul_bcast_rows_init_short_with_dt(icb0, icb1)` | reconfig + `mul_bcast_rows_init_short(icb0, icb1)` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` |
| `sub_bcast_cols_init_short_with_dt(icb0, icb1)` | reconfig + `sub_bcast_cols_init_short(icb0, icb1)` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` |
| `mul_tiles_bcast_scalar_init_short_with_dt(icb0, icb1)` | reconfig + `mul_tiles_bcast_scalar_init_short(icb0, icb1)` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` |
| `sub_tiles_bcast_scalar_init_short_with_dt(icb0, icb1)` | reconfig + `sub_tiles_bcast_scalar_init_short(icb0, icb1)` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` |

### Convenience "to_cb" Helpers

These perform a complete compute step: wait inputs, acquire, compute, commit, wait, pack, release, pop, push.

| Helper | Operation | Signature |
|--------|-----------|-----------|
| `mul_tiles_to_cb` | `ocb = icb0 * icb1` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` |
| `add_tiles_to_cb` | `ocb = icb0 + icb1` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` |
| `sub_tiles_to_cb` | `ocb = icb0 - icb1` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` |
| `copy_tile_to_cb` | `ocb = icb` | `(icb, ocb, itile=0, pop=1)` |
| `recip_tile_to_cb` | `ocb = 1/icb` | `(icb, ocb, itile=0, pop=1)` |
| `mul_tiles_bcast_rows_to_cb` | `ocb = icb0 *_bcast_rows icb1` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` |
| `mul_tiles_bcast_cols_to_cb` | `ocb = icb0 *_bcast_cols icb1` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` |
| `sub_tiles_bcast_cols_to_cb` | `ocb = icb0 -_bcast_cols icb1` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` |
| `sub_tiles_bcast_rows_to_cb` | `ocb = icb0 -_bcast_rows icb1` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` |

Note: The `pop` parameters control whether to pop front from each input CB after the operation. Setting `pop=0` allows the tile to remain for reuse.

---

## Normalization Kernel Utilities (kernel_util/) -- Reference for layer_norm_rm

The repository contains reusable normalization compute utilities under `ttnn/cpp/ttnn/operations/normalization/kernel_util/`. These are highly relevant for layer_norm_rm:

### `numeric.h` -- Row-Wise Accumulation and Mean

**`row_wise_accumulate_with_epilogue`**: Generic reduce-then-epilogue-then-pack pattern.
```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, bool FLOAT32_REDUCTION,
          typename input_policy, WaitAtEndPolicy wait_at_end_policy, typename Epilogue, typename... AdditionalCBs>
void row_wise_accumulate_with_epilogue(
    uint32_t cb_in, uint32_t cb_scalar, uint32_t cb_out,
    uint32_t num_tiles, uint32_t block_size, uint32_t N,
    Epilogue epilogue, AdditionalCBs... additional_cbs);
```

Internally:
1. Waits for scalar tiles (1 or 2 if partial last tile)
2. `tile_regs_acquire()`
3. Calls `reduce_init<reduce_type, reduce_dim, FLOAT32_REDUCTION>(cb_in, cb_scalar, cb_out)`
4. Loops over blocks of tiles calling `reduce_tile` for each
5. Calls `reduce_uninit<FLOAT32_REDUCTION>()`
6. Calls `epilogue()` (user-provided, e.g., multiply by 1/N for mean)
7. `tile_regs_commit()`, `tile_regs_wait()`, pack, release
8. Optionally `cb_wait_front(cb_out, 1)` at end

**`row_wise_mean`**: Specialization that computes mean by using `1/N` scalar multiply as epilogue.
```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, bool FLOAT32_REDUCTION,
          typename input_policy, WaitAtEndPolicy wait_at_end_policy>
void row_wise_mean(uint32_t cb_in, uint32_t cb_scalar, uint32_t cb_out,
                   uint32_t N, uint32_t num_tiles, uint32_t block_size);
```

Epilogue: `detail::scale_dest(dst0, bit_cast<uint32_t>(1.0f / N))` which calls `binop_with_scalar_tile_init()` + `mul_unary_tile(dst0, 1/N_as_u32)`.

**`row_wise_mean_with_pre_add`**: Like `row_wise_mean` but accumulates from two CBs.

### `policies.h` -- Input Handling Policies

| Policy | Pop? | Sync Full Block? | Use Case |
|--------|------|------------------|----------|
| `PartialBlockWithPopPolicy` | Yes | No | Streaming: pop after partial block |
| `PartialBlockWithoutPopPolicy` | No | No | Reuse: keep data for second pass (e.g., centering) |
| `FullBlockWithPopPolicy` | Yes | Yes | Wait for entire block before processing |
| `FullBlockWithoutPopPolicy` | No | Yes | Reuse with full block sync |

**Critical for layer_norm_rm**: Use `PartialBlockWithoutPopPolicy` when computing mean (first pass) so tiles remain for the centering pass.

### `blocked_range.h` -- Block Iteration Utility

`blocks(total, block_size)` creates an iterable range of `Block` objects. Each block knows its start, end, size, and provides `local()` and `global()` iterators.

### `combine_welford.h` -- Welford Partial Combination

For distributed layer norm (not directly needed for single-device layer_norm_rm), but shows the pattern for combining partial mean/variance results.

### `layernorm_compute_utils.h` -- Tilize/Untilize Helpers

**Critical for row-major layer_norm_rm**:

**`tilize_all_blocks_to_cb<block_size>(cb_in_rm, cb_in, Wt)`**: Converts row-major input to tile layout.
```cpp
reconfig_data_format(cb_in_rm, cb_in_rm);
pack_reconfig_data_format(cb_in);
tilize_init(cb_in_rm, block_size, cb_in);
for (auto block : blocks(Wt, block_size)) {
    cb_wait_front(cb_in_rm, block.full_block_size());
    cb_reserve_back(cb_in, block.full_block_size());
    tilize_block(cb_in_rm, block.full_block_size(), cb_in);
    cb_push_back(cb_in, block.full_block_size());
    cb_pop_front(cb_in_rm, block.full_block_size());
}
tilize_uninit(cb_in_rm, cb_in);
```

**`untilize_all_blocks_from_cb<block_size>(cb_out, cb_out_rm, Wt)`**: Converts tile layout to row-major output.
```cpp
reconfig_data_format(cb_out, cb_out);
pack_untilize_init<block_size, block_size>(cb_out, cb_out_rm);
for (auto block : blocks(Wt, block_size)) {
    cb_wait_front(cb_out, block.full_block_size());
    cb_reserve_back(cb_out_rm, block.full_block_size());
    pack_untilize_block<block_size, block_size>(cb_out, 1, cb_out_rm);
    cb_push_back(cb_out_rm, block.full_block_size());
    cb_pop_front(cb_out, block.full_block_size());
}
pack_untilize_uninit(cb_out_rm);
```

**Important**: After `tilize_uninit`, you must re-initialize the compute pipeline with `binary_op_init_common` since tilize reconfigures hardware state.

---

## Mapping Batch Norm Patterns to Layer Norm RM Requirements

### What batch_norm provides as reference:
1. **Compute pipeline pattern**: tile_regs_acquire/commit/wait/release cycle
2. **Binary op chaining**: sub_tiles + binary_dest_reuse_tiles for (x-mean)*inv_std
3. **SFPU rsqrt pattern**: add_tiles + rsqrt_tile for 1/sqrt(var+eps)
4. **Broadcast reuse pattern**: CB wait_front before loop, pop_front after loop
5. **CB aliasing for conditional output routing**: cb_affine_or_out / cb_scaled_output
6. **Optional affine (gamma/beta) pattern**: conditional mul + add

### What layer_norm_rm needs additionally:
1. **Reduction operations**: `reduce_init/reduce_tile/reduce_uninit` to compute row-wise sum for mean and variance (batch_norm receives these pre-computed). Use the `numeric.h` utilities.
2. **Multi-pass over same data**: First pass computes mean, second pass computes (x-mean)^2 for variance, third pass normalizes. Need `PartialBlockWithoutPopPolicy` for first pass to keep tiles.
3. **Tilize/untilize**: Since input is row-major, need `tilize_all_blocks_to_cb` before compute and `untilize_all_blocks_from_cb` after. Use `layernorm_compute_utils.h`.
4. **Scalar tile for reduce**: Need a CB with tile of 1's for `reduce_tile`'s scaler parameter. May need two scalar tiles if last tile is partial.
5. **Scalar multiply for mean**: `binop_with_scalar_tile_init()` + `mul_unary_tile(dst, 1/W_as_u32)` to divide sum by width.

---

## Kernel Implementations Summary

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM (input) | CB_c0, CB_c4 | Read input tiles, fill epsilon tile |
| compute (FPU-path: batch_norm_kernel) | RISCV_2 | N/A | CB_c0,c1,c3,c4,c5,c6 | CB_c2,c7,c8 | sub, mul, add, rsqrt, binary_dest_reuse |
| compute (SFPU-path: batch_norm_sfpu_kernel) | RISCV_2 | N/A | CB_c0,c1,c3,c4,c5,c6 | CB_c2,c7,c8 | copy_tile, add/sub/mul_binary_tile, rsqrt |
| writer_batch_norm | RISCV_1 | NOC1 | DRAM (mean/var/weight/bias) | CB_c1,c3,c5,c6 + writes CB_c2 to DRAM | Read params, broadcast fill, write output |

---

## Implementation Notes

1. **Two kernel variants**: The FPU-path uses `eltwise_binary.h` APIs (add_tiles, sub_tiles, mul_tiles) which operate directly on CB tiles via the unpacker. The SFPU-path uses explicit `copy_tile` to DST followed by `add_binary_tile`/`sub_binary_tile`/`mul_binary_tile`. Both exist to support FP32 accumulation mode.

2. **binary_dest_reuse_tiles optimization**: In the FPU path, the subtraction result (x-mean) in DST is immediately multiplied by inv_std without an intermediate pack/unpack. This avoids a CB round-trip per tile.

3. **CB aliasing is compile-time in practice**: Even though `cb_affine_or_out` and `cb_scaled_output` are computed at runtime from `weight_has_value` and `bias_has_value`, these booleans come from compile-time args, so the compiler can optimize the branches.

4. **All CBs are double-buffered (capacity=2)**: This enables overlap between reader/compute and compute/writer but the actual utilization depends on the blocking behavior of wait_front/reserve_back.

5. **Channel-frequency tracking**: The `freq`/`counter` mechanism allows a core to handle tiles that span channel boundaries. The `tile_start` parameter tells the compute kernel where within a channel the first tile falls.

6. **Epsilon is broadcast-filled differently from other parameters**: Epsilon uses `fill_with_val` (writes the same value to every element) in the reader, while mean/var/weight/bias use `FILL_TILE_WITH_FIRST_ELEMENT` (reads element[0] of a DRAM tile and broadcasts it) in the writer.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do tile_regs_acquire/commit/wait/release work?"
   **Reason**: Needed to understand the DST register synchronization protocol between Math and Pack cores.
   **Key Findings**: These form a 4-phase pipeline: acquire (Math claims DST) -> commit (Math hands off to Pack) -> wait (Pack waits for data) -> release (Pack frees DST). Must always be called in this order, even if no packing occurs.

2. **Query**: "How do binary compute operations work (add_tiles, sub_tiles, mul_tiles, binary_dest_reuse_tiles)?"
   **Reason**: Core compute operations used throughout the kernel.
   **Key Findings**: FPU binary ops take two CB IDs and tile indices, writing result to DST. `binary_dest_reuse_tiles` with `DEST_TO_SRCA` uses the current DST value as operand A, avoiding intermediate pack/unpack for chained operations.

3. **Query**: "How does rsqrt_tile work? What SFPU ops are available for normalization?"
   **Reason**: The rsqrt operation is central to normalization (computing inv_std).
   **Key Findings**: `rsqrt_tile_init()` + `rsqrt_tile(dst_index)` computes 1/sqrt(x) in-place on a DST register. Available SFPU ops: rsqrt_tile, recip_tile, sqrt_tile. The SFPU path requires explicit copy_tile to move data into DST registers before operations.

4. **Query**: "What is the experimental::CircularBuffer API and how do binary_op_init_common/unary_op_init_common work?"
   **Reason**: Understanding CB synchronization and compute pipeline initialization.
   **Key Findings**: CircularBuffer wraps cb_wait_front/pop_front/reserve_back/push_back. binary_op_init_common initializes all three threads (UNPACK, MATH, PACK) for binary operations from two input CBs to one output CB.

5. **Query**: "How do reduce_init, reduce_tile, reduce_uninit work?"
   **Reason**: Layer_norm_rm will need reduce operations for computing mean and variance, which batch_norm does not use.
   **Key Findings**: `reduce_init<PoolType, ReduceDim, FP32>(cb_in, cb_scalar, cb_out)` configures the hardware for reduction. `reduce_tile(cb_in, cb_scalar, itile, scaler_tile_idx, dst)` accumulates one tile into DST. The scalar tile should contain 1's for PoolType::SUM. For partial last tiles, a second scalar tile with masked values is needed.

6. **Query**: "What is binary_dest_reuse_tiles and how do binop_with_scalar_tile_init/mul_unary_tile/add_unary_tile work?"
   **Reason**: Understanding scalar operations for the mean computation (divide by N).
   **Key Findings**: `mul_unary_tile(dst, scalar_as_u32)` multiplies all elements in a DST tile by a scalar. The scalar must be bit-cast from float to uint32_t. `binop_with_scalar_tile_init()` must be called before these operations.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Provides the `_with_dt` wrapper functions and convenience `_to_cb` helpers used in the compute kernel.
   **Key Information**: All wrappers conditionally call `reconfig_data_format()` when `FP32_DEST_ACC_EN` is defined. The `_to_cb` helpers encapsulate the full acquire-compute-commit-wait-pack-release cycle for common operations.

2. **Source**: `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h`
   **Reason**: Contains reusable row-wise accumulation and mean computation functions that layer_norm_rm should use.
   **Key Information**: `row_wise_mean` and `row_wise_accumulate_with_epilogue` handle reduce operations with configurable policies for CB pop behavior and block synchronization. Uses `reduce_init/reduce_tile/reduce_uninit` internally.

3. **Source**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_compute_utils.h`
   **Reason**: Contains tilize/untilize helpers needed for row-major input/output.
   **Key Information**: `tilize_all_blocks_to_cb` and `untilize_all_blocks_from_cb` handle row-major to tile format conversion. After tilize_uninit, binary_op_init_common must be re-called.

4. **Source**: `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/policies.h`
   **Reason**: Input handling policies for controlling whether tiles are popped after reduction passes.
   **Key Information**: `PartialBlockWithoutPopPolicy` (pop=false) is critical for multi-pass algorithms where the same input tiles are needed again (e.g., first pass for mean, second pass for centering).

5. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Contains the fill functions used for epsilon and broadcast parameter tiles.
   **Key Information**: `fill_with_val_bfloat16` fills 512 uint32 words, `fill_with_val<1024, float>` fills 1024 float words. `fill_tile_with_first_element_bfloat16` reads element[0] and broadcasts to all 1024 positions.
