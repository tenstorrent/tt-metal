# Batch Norm Implementation Analysis

## Overview

Batch normalization computes:
```
output = (input - batch_mean) / sqrt(batch_var + eps)  [* weight]  [+ bias]
```

The operation receives **pre-computed** batch_mean and batch_var tensors (it does NOT compute the mean/variance itself). It normalizes the input per-channel, where mean and variance are scalar-per-channel values broadcast across HxW spatial dimensions. Optional affine parameters (weight, bias) are also per-channel.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Two compute kernel variants**:
- `batch_norm_kernel.cpp` -- FPU-based (non-fp32 accumulation path), uses `binary_op_init_common` and FPU binary operations with `binary_dest_reuse_tiles` pattern.
- `batch_norm_sfpu_kernel.cpp` -- SFPU-based (fp32 destination accumulation path), uses `unary_op_init_common` and explicit `copy_tile`+SFPU binary operations (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`).

The variant is selected at program creation time based on `fp32_dest_acc_en`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` = total output tiles |
| **Loop structure** | Tiles iterated in N-C-HtWt order; broadcast parameters (mean, var, weight, bias) reloaded once per channel change |

The compute kernel processes tiles in a pattern governed by `freq` (= cHt * cWt, the number of tiles per channel's HxW plane) and `counter` (= start_tile_id % freq, the initial offset within a channel). This naturally groups tiles by channel for parameter reuse.

## Tensor Format and Layout

### Input Tensors

| Property | input (x) | batch_mean | batch_var | weight (optional) | bias (optional) |
|----------|-----------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] or [N, C, 1, 1] | same as batch_mean | same as batch_mean | same as batch_mean |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (via TensorAccessor) | DRAM (via TensorAccessor) | DRAM (via TensorAccessor) | DRAM (via TensorAccessor) | DRAM (via TensorAccessor) |
| **Data type** | BFLOAT16 or FLOAT32 | same | same | same | same |

### Output Tensor

| Property | output |
|----------|--------|
| **Logical shape** | same as input [N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Data type** | same as input or configured via `dtype` attribute |

### Layout Transformations

**No tilize/untilize** -- all tensors are already in TILE_LAYOUT.

**Broadcast via `FILL_TILE_WITH_FIRST_ELEMENT`**: The batch_mean, batch_var, weight, and bias tensors have shapes like [N, C, 1, 1]. After reading a tile from these tensors (which contains the scalar value in position [0,0]), the writer kernel calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast this scalar across the entire 32x32 tile. This is a **data-movement-side broadcast** -- it happens in L1 before the tile is pushed to the compute kernel. The compute kernel then performs element-wise operations with these broadcast tiles.

## Data Flow Pattern

### High-Level Flow

```
Reader (RISC-V 0):
  1. Fill eps CB once (scalar broadcast to full tile)
  2. For each output tile: read input tile from DRAM -> CB c_0

Writer (RISC-V 1):
  For each channel:
    3. Read batch_mean tile from DRAM -> fill_tile_with_first_element -> CB c_1
    4. Read batch_var tile from DRAM -> fill_tile_with_first_element -> CB c_3
    5. [If weight]: Read weight tile -> fill_tile_with_first_element -> CB c_5
    6. [If bias]: Read bias tile -> fill_tile_with_first_element -> CB c_6
    For each HtWt tile in this channel:
      7. Wait for compute output in CB c_2 -> write to DRAM

Compute (RISC-V 2):
  8. Wait for eps in CB c_4 (once, persists entire program)
  For each channel group:
    9.  Wait for batch_var (CB c_3) + eps (CB c_4) -> add -> rsqrt -> CB c_7 (den)
    10. Wait for batch_mean (CB c_1) and den (CB c_7)
    11. [Wait for weight (CB c_5) if present]
    12. [Wait for bias (CB c_6) if present]
    For each HtWt tile:
      13. Wait for input (CB c_0)
      14. sub_tiles(input, batch_mean) -> (input - mean)
      15. mul with den -> (input - mean) * rsqrt(var + eps)   [-> CB c_8 or c_2]
      16. [If weight]: mul with weight                         [-> CB c_8 or c_2]
      17. [If bias]: add bias                                  [-> CB c_2]
    18. Pop batch_mean, den, [weight], [bias]
```

### Key Observation: Writer Reads, Reader Reads

The **writer kernel** reads batch_mean, batch_var, weight, and bias from DRAM (and writes the output). The **reader kernel** reads only the input tensor and fills the eps constant. This is a **split reader pattern** where data ingestion responsibilities are divided between reader and writer to balance NoC utilization (reader uses NoC0, writer uses NoC1).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | input_tensor_cb | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | batch_mean_tensor_cb | Batch mean (broadcast) | 2 tiles | 1 tile | Double | Writer | Compute | Channel (reused across HtWt tiles in one channel) |
| c_2 | output_tensor_cb | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | batch_var_tensor_cb | Batch variance (broadcast) | 2 tiles | 1 tile | Double | Writer | Compute | Block (consumed once per channel to produce den) |
| c_4 | eps_cb | Epsilon constant (broadcast) | 2 tiles | 1 tile | Double | Reader | Compute | Program (filled once, popped at program end) |
| c_5 | weight_tensor_cb | Weight (broadcast, optional) | 2 tiles | 1 tile | Double | Writer | Compute | Channel (reused across HtWt tiles in one channel) |
| c_6 | bias_tensor_cb | Bias (broadcast, optional) | 2 tiles | 1 tile | Double | Writer | Compute | Channel (reused across HtWt tiles in one channel) |
| c_7 | den_cb | Intermediate: 1/sqrt(var+eps) | 2 tiles | 1 tile | Double | Compute | Compute | Channel (produced and consumed within compute for one channel group) |
| c_8 | temp_1_cb | Intermediate: normalized result before final output | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile, intermediate between affine transform steps) |

**All CBs use capacity = 2 tiles, block size = 1 tile (double-buffered)**.

### CB Routing Based on Weight/Bias Presence

The compute kernel dynamically routes output to different CBs based on whether weight/bias are present:

```
cb_affine_or_out = (weight_has || bias_has) ? cb_tmp_1 (c_8) : cb_output_0 (c_2)
cb_scaled_output = (bias_has)               ? cb_tmp_1 (c_8) : cb_output_0 (c_2)
```

This means:
- **No weight, no bias**: normalized result goes directly to c_2 (output)
- **Weight only**: normalized -> c_8, then weight*c_8 -> c_2
- **Bias only**: normalized -> c_8, then c_8+bias -> c_2
- **Weight and bias**: normalized -> c_8, weight*c_8 -> c_8, c_8+bias -> c_2

The c_8 (temp_1) CB serves as a ping-pong intermediate between affine transform phases.

## Pipeline Pattern Summary

All CBs are configured with capacity=2 tiles and block_size=1 tile, enabling **double-buffering** throughout. This allows overlap between:
- Reader producing input tiles and compute consuming them
- Compute producing output tiles and writer draining them
- Writer producing parameter tiles and compute consuming them

## Multi-Pass Data Reuse Patterns

This is the critical design pattern for understanding batch_norm's compute structure:

### Epsilon CB (c_4) -- Program Lifetime
- **Filled once** by reader at program start via `fill_with_val` (scalar broadcast to full tile)
- Compute does `cb_wait_front(cb_eps, 1)` once before the main loop
- **Never popped during the loop** -- persists across ALL channel iterations
- `cb_pop_front(cb_eps, 1)` called once at program end
- This is the simplest reuse pattern: fill once, use many times

### Batch Mean (c_1), Weight (c_5), Bias (c_6) -- Channel Lifetime
- Writer pushes one tile per channel, after `FILL_TILE_WITH_FIRST_ELEMENT` broadcast
- Compute does `cb_wait_front` once at the start of each channel group (outer iteration)
- These CBs are **held open** (not popped) for the entire inner loop over HtWt tiles
- Popped once at the end of each channel group iteration
- Pattern: `wait_front -> [use N times in inner loop] -> pop_front`

### Batch Variance (c_3) -- Consumed Once Per Channel
- Writer pushes one tile per channel
- Compute immediately consumes it to produce `den` in c_7
- Popped right after producing den (before the inner HtWt loop)

### Den CB (c_7) -- Channel Lifetime, Compute-Internal
- **Produced by compute** (not by any dataflow kernel)
- Produced once per channel from batch_var + eps
- Held open for the inner HtWt loop (used in every multiply)
- Popped at end of channel group, alongside batch_mean

### Input (c_0) -- Per-Tile Lifetime
- Each input tile is pushed by reader, consumed by compute, and popped immediately per tile

### Output (c_2) -- Per-Tile Lifetime
- Each output tile is produced by compute and drained by writer per tile

## Scalar/Constant CB Setup

### Epsilon (eps) Setup in Reader

```cpp
// Reader kernel:
const auto eps = get_arg_val<uint32_t>(0);  // packed scalar from runtime args
union { float f; uint32_t u; } scalar;
scalar.u = eps;
cb_reserve_back(cb_id_eps, onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    FILL_WITH_VALUE_FLOAT(cb_id_eps, scalar.f);  // float32 path: fills 1024 floats
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_eps, eps);              // bfloat16 path: fills 512 uint32 (packed pairs)
#endif
cb_push_back(cb_id_eps, onetile);
```

The epsilon scalar is packed on the host side:
```cpp
// Program factory:
const auto packed_scalar_eps = input_tensor.dtype() == DataType::FLOAT32
    ? std::bit_cast<uint32_t>(scalar)
    : pack_two_bfloat16_into_uint32({scalar, scalar});
```

For bfloat16: the scalar is double-packed into a uint32 (`{eps, eps}`), and `fill_with_val_bfloat16` writes 512 uint32 values to fill 1024 bfloat16 elements (one full tile).

For float32: the scalar is reinterpreted as uint32 for transport, then cast back to float in the kernel, and `fill_with_val<1024, float>` writes 1024 float values.

### Parameter Broadcast in Writer

```cpp
// Writer kernel (for each channel):
noc_async_read_tile(tile_offset, src, l1_write_addr);  // read [N,C,1,1] tile from DRAM
noc_async_read_barrier();
FILL_TILE_WITH_FIRST_ELEMENT(cb_id_src);  // broadcast element[0,0] to all 1024 positions
cb_push_back(cb_id_src, onetile);
```

`FILL_TILE_WITH_FIRST_ELEMENT` reads the first element of the tile (which holds the scalar mean/var/weight/bias for that channel) and fills the entire tile with that value. This is a purely L1-side operation that happens after the NoC read completes and before the tile is pushed to the compute kernel.

## Index Calculations

### Output Tile Linearization

Total output tiles = `N * C * Ht * Wt` where `Ht = H/tile_height`, `Wt = W/tile_width`.

Tiles are distributed to cores using `split_work_to_cores` which assigns contiguous ranges of linear tile IDs. Each core gets `[start_tile_id, start_tile_id + num_tiles_per_core)`.

### Channel Frequency and Counter (Compute Kernel)

```cpp
// Program factory:
auto counter = start_tile_id % cHtWt;  // position within the current channel
auto freq = cHtWt;                       // tiles per channel (Ht * Wt)

// Compute kernel:
uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
```

- `freq` (= `cHt * cWt`): number of spatial tiles per channel. This defines the "broadcast group size" -- how many input tiles share the same mean/var/weight/bias values.
- `counter` (= `start_tile_id % freq`): the starting offset within a channel. If a core's tile range starts mid-channel, this tells the compute kernel to start the inner loop at that offset.
- `complete_iterations`: number of full channel groups this core processes.
- `remaining_iterations`: leftover tiles in a partial channel at the end.

The `batchnorm_bcast_tiles` function handles one channel group. It receives `tile_start` as the initial offset (non-zero for the first partial group) and `freq` as the upper bound (or `remaining_iterations` for the last partial group).

### Reader Index Calculation

```cpp
// Reader iterates N, C, HtWt in nested loops:
uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;
// n_stride = aHt * aWt * aC * (aN > 1)  -- stride between batches
// c_stride = aHt * aWt * (aC > 1)       -- stride between channels
// Advances tile_offset by 1 for each spatial tile, adds next_channel_shift at channel boundary, next_batch_shift at batch boundary
```

### Writer Index Calculation

Similar N-C structure. The writer reads parameters at channel granularity:
```cpp
uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
// For each channel: read mean, var, [weight], [bias] at tile_offset
// Then iterate HtWt output tiles, writing to linear output positions
```

## Memory Access Patterns

### Read Pattern
- **Input tensor (reader)**: Sequential tile reads within each HtWt group, with stride jumps at channel and batch boundaries. Uses TensorAccessor for address calculation.
- **Parameters (writer)**: One tile read per channel. The tile at the (n, c) position in the parameter tensor is read, then broadcast via `fill_tile_with_first_element`.
- **Epsilon**: Written once to L1, never re-read from DRAM.

### Write Pattern
- **Output (writer)**: Sequential tile writes using linear tile ID `start_tile_id + num_tiles_written`. Uses TensorAccessor.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (row-major enumeration) |
| **Grid dimensions** | compute_with_storage_grid_size.x * compute_with_storage_grid_size.y |
| **Total cores** | Up to full device grid (e.g., 8x8 = 64 on Wormhole) |
| **Work per core** | `num_output_tiles / num_cores` tiles (ceil for group 1, floor for group 2) |
| **Load balancing** | Two-group split via `split_work_to_cores`: group 1 gets `ceil(tiles/cores)` tiles, group 2 gets `floor(tiles/cores)` tiles |

Cores not in either group receive zero-filled runtime args and return immediately (`if (num_tiles == 0) return`).

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if weight tensor present, 0 otherwise. Controls affine multiply branch. |
| 1 | bias_has_value | uint32_t | 1 if bias tensor present, 0 otherwise. Controls affine add branch. |
| 2 | cb_input | uint32_t | CB index for input tensor (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon constant (c_4) |
| 7 | cb_den | uint32_t | CB index for 1/sqrt(var+eps) intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for weight (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for temp intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB index for bias (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total number of output tiles for this core |
| 1 | tile_freq | uint32_t | Tiles per channel group (cHt * cWt). Defines broadcast reuse period. |
| 2 | tile_start | uint32_t | Starting offset within the first channel group (start_tile_id % freq) |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon packed as uint32 (bit_cast for f32, double-packed bf16) |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | First output tile ID for this core |
| 3 | num_tiles | uint32_t | Number of tiles to process |
| 4 | HtWt | uint32_t | cHt * cWt (tiles per channel spatial plane) |
| 5 | n_stride | uint32_t | Tile stride between batches (0 if N==1) |
| 6 | c_stride | uint32_t | Tile stride between channels (0 if C==1) |
| 7 | N | uint32_t | Output batch dimension |
| 8 | C | uint32_t | Output channel dimension |
| 9 | Ht | uint32_t | Output height in tiles |
| 10 | Wt | uint32_t | Output width in tiles |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Batch mean buffer address |
| 1 | batch_var_addr | uint32_t | Batch variance buffer address |
| 2 | weight_addr | uint32_t | Weight buffer address (0 if not present) |
| 3 | bias_addr | uint32_t | Bias buffer address (0 if not present) |
| 4 | dst_addr | uint32_t | Output buffer address |
| 5 | start_tile_id | uint32_t | First output tile ID for this core |
| 6 | num_tiles | uint32_t | Number of tiles to process |
| 7 | HtWt | uint32_t | Tiles per channel spatial plane |
| 8 | n_stride | uint32_t | Tile stride between batches (parameter tensor) |
| 9 | c_stride | uint32_t | Tile stride between channels (parameter tensor) |
| 10 | N | uint32_t | Output batch dimension |
| 11 | C | uint32_t | Output channel dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |

## Kernel Implementations

### Kernel Specification Table

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM (input tensor) | c_0 (input), c_4 (eps) | Read input tiles; fill eps constant tile |
| writer_batch_norm | RISCV_1 | NOC1 | DRAM (mean, var, weight, bias); c_2 (output from compute) | c_1 (mean), c_3 (var), c_5 (weight), c_6 (bias); DRAM (output) | Read parameters with scalar broadcast; write output tiles |
| batch_norm_kernel | RISCV_2 (compute) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2 (output), c_7 (den), c_8 (temp) | sub, add, rsqrt, mul (FPU path) |
| batch_norm_sfpu_kernel | RISCV_2 (compute) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2 (output), c_7 (den), c_8 (temp) | copy_tile, add/sub/mul_binary_tile, rsqrt_tile (SFPU path) |

### Compute Kernel: FPU Variant (`batch_norm_kernel.cpp`)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`

**Includes**:
- `api/compute/eltwise_binary.h` -- provides `sub_tiles`, `add_tiles`, `mul_tiles`, `binary_dest_reuse_tiles`, `binary_op_init_common`
- `ttnn/kernel/compute/moreh_common.hpp` -- provides `pack_tile_with_dt`, `add_tiles_init_with_dt`, `mul_tiles_init_with_dt`, `sub_tiles_init_with_dt`

**Initialization**:
```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```
Signature: `binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb)`
Configures unpack hardware for AB mode, math hardware, and pack hardware. Sets up the compute pipeline for two-input binary operations.

**Phase 1: Compute den = 1/sqrt(var + eps)** (once per channel)

```cpp
// Acquire DST regs
tile_regs_acquire();

// Step 1: add var + eps
add_tiles_init_with_dt(cb_batch_var, cb_eps);        // reconfigure data formats + init add
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);         // dst0 = batch_var[0] + eps[0]

// Step 2: rsqrt in-place
rsqrt_tile_init();                                    // init SFPU rsqrt
rsqrt_tile(dst0);                                     // dst0 = 1/sqrt(dst0)

tile_regs_commit();  // signal math done
tile_regs_wait();    // wait for pack to be ready
pack_tile_with_dt(dst0, cb_den);                      // pack dst0 -> cb_den (c_7)
tile_regs_release();
```

API signatures used:
- `add_tiles_init_with_dt(uint32_t icb0, uint32_t icb1)` -- wraps `reconfig_data_format` (if FP32_DEST_ACC_EN) + `add_tiles_init`
- `add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` -- unpacks tiles from icb0 and icb1, adds them, result in DST[idst]
- `rsqrt_tile_init()` -- initializes SFPU for reciprocal square root
- `rsqrt_tile(uint32_t idst)` -- computes 1/sqrt(DST[idst]) in-place
- `pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb)` -- wraps `pack_reconfig_data_format` (if FP32_DEST_ACC_EN) + `pack_tile`

**Phase 2: For each HtWt tile -- normalize + optional affine**

```cpp
// Step 3: subtract mean
tile_regs_acquire();
sub_tiles_init(cb_other, cb_bcast);
sub_tiles(cb_other, cb_bcast, 0, 0, 0);              // dst0 = input[0] - batch_mean[0]

// Step 4: multiply by den (dest reuse pattern!)
binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
// Result: dst0 = (input - mean) * den

tile_regs_commit();
tile_regs_wait();
pack_tile_with_dt(0, cb_affine_or_out);               // pack to c_8 or c_2 depending on affine
tile_regs_release();
```

**The `binary_dest_reuse_tiles` Pattern** (critical for understanding chained operations):

```cpp
template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType binary_reuse_dest>
ALWI void binary_dest_reuse_tiles_init(uint32_t icb0);

template <EltwiseBinaryType eltwise_binary_type, EltwiseBinaryReuseDestType binary_reuse_dest>
ALWI void binary_dest_reuse_tiles(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index);
```

With `DEST_TO_SRCA`: The tile currently in DST[dst_tile_index] is moved to SRCA, the tile from `in_cb_id` goes to SRCB, and the binary operation result goes back to DST[dst_tile_index]. This allows **chaining operations without intermediate pack/unpack**: the subtraction result stays in DST and is directly used as an operand for the multiplication.

This saves one pack+unpack cycle compared to writing the subtraction result to a CB and reading it back.

**Phase 3: Optional weight multiply** (if weight_has_value)

```cpp
// Wait for intermediate from c_8, multiply by weight
mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);   // dst0 = normalized * weight
pack_tile_with_dt(dst0, cb_scaled_output);             // -> c_8 or c_2
```

**Phase 4: Optional bias add** (if bias_has_value)

```cpp
// Wait for intermediate from c_8, add bias
add_tiles_init_with_dt(cb_tmp_1, cb_bias);
add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);             // dst0 = (normalized * weight) + bias
pack_tile_with_dt(dst0, cb_output_0);                  // -> c_2 (always final output)
```

### Compute Kernel: SFPU Variant (`batch_norm_sfpu_kernel.cpp`)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`

**Includes**:
- `api/compute/eltwise_binary_sfpu.h` -- provides `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile` and their `_init` counterparts
- `api/compute/eltwise_unary/eltwise_unary.h`, `rsqrt.h`, `sfpu_split_includes.h`
- `ttnn/kernel/compute/moreh_common.hpp`

**Initialization**:
```cpp
unary_op_init_common(cb_other, cb_output_0);
```
Uses unary init (not binary) because SFPU binary ops use `copy_tile` to load operands into DST, then SFPU operations on DST registers.

**Key Difference from FPU Variant**: Instead of using the FPU's binary operation pipeline (which unpacks from two CBs), the SFPU variant:
1. Uses `copy_tile` to load tiles from CBs into specific DST register slots
2. Uses `add_binary_tile(idst0, idst1, odst)`, `sub_binary_tile(idst0, idst1, odst)`, `mul_binary_tile(idst0, idst1, odst)` which operate entirely within DST registers

This approach is necessary for FP32 destination accumulation because the SFPU binary operations preserve full fp32 precision in DST.

**Phase 1 (SFPU): Compute den**
```cpp
tile_regs_acquire();
tile_regs_wait();
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
copy_tile(cb_batch_var, 0, 0);                     // DST[0] = batch_var

add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
copy_tile(cb_eps, 0, 1);                           // DST[1] = eps
add_binary_tile(0, 1, 0);                          // DST[0] = DST[0] + DST[1]

rsqrt_tile_init();
rsqrt_tile(0);                                     // DST[0] = 1/sqrt(DST[0])
pack_tile(0, cb_den);                              // pack to c_7
tile_regs_commit();
tile_regs_release();
```

API signatures:
- `copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose = 0)` -- reconfigures data format from old_cbid to new_cbid, then calls `copy_tile_to_dst_init_short(new_cbid)`. The old_cbid parameter tells the hardware what the previous source format was so it can reconfigure.
- `copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)` -- copies tile from CB to DST register
- `add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` -- SFPU binary add: DST[odst] = DST[idst0] + DST[idst1]
- `sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` -- SFPU binary sub: DST[odst] = DST[idst0] - DST[idst1]
- `mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` -- SFPU binary mul: DST[odst] = DST[idst0] * DST[idst1]

**Phase 2 (SFPU): Normalize each tile**
```cpp
copy_tile(cb_other, 0, 0);                         // DST[0] = input
sub_binary_tile_init();
copy_tile(cb_bcast, 0, 1);                         // DST[1] = batch_mean
sub_binary_tile(0, 1, 0);                          // DST[0] = input - mean

mul_binary_tile_init();
copy_tile(cb_den, 0, 1);                           // DST[1] = den
mul_binary_tile(0, 1, 0);                          // DST[0] = (input - mean) * den
pack_tile(0, cb_affine_or_out);
```

Note the SFPU variant uses `pack_tile` directly (not `pack_tile_with_dt`) in most places since fp32 accumulation mode means the pack format is already configured correctly for the output CB.

**Phase 3/4 (SFPU): Optional affine** -- follows same pattern as FPU variant but using `copy_tile` + `mul_binary_tile` / `add_binary_tile`.

### Reader Kernel (`reader_batch_norm.cpp`)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp`

**Summary**: Reads input tiles from DRAM in N-C-HtWt order using TensorAccessor. Fills epsilon CB once at start. Simple sequential tile reads with noc_async_read_barrier per tile.

### Writer Kernel (`writer_batch_norm.cpp`)

**File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`

**Summary**: Reads batch_mean, batch_var, weight, and bias from DRAM per-channel, broadcasts each to full tile via `FILL_TILE_WITH_FIRST_ELEMENT`. Writes output tiles to DRAM. Uses TensorAccessor for all address calculations.

## Implementation Notes

### Two Compute Variants: When to Use Which

The choice between FPU and SFPU variants is made based on `fp32_dest_acc_en`:
- **fp32_dest_acc_en = false**: Uses `batch_norm_kernel.cpp` (FPU path). Binary operations use the FPU's native unpack-A/unpack-B pipeline. The `binary_dest_reuse_tiles` pattern enables chaining without intermediate CB round-trips.
- **fp32_dest_acc_en = true**: Uses `batch_norm_sfpu_kernel.cpp` (SFPU path). All binary operations are done via SFPU, which preserves fp32 precision in DST. Uses explicit `copy_tile` to load operands.

The SFPU variant also sets `UnpackToDestMode::UnpackToDestFp32` for all relevant CBs, ensuring tiles are unpacked to fp32 format in the destination registers.

### Key Design Pattern: Channel-Grouped Broadcast

The `freq`/`counter` mechanism in the compute kernel is the core pattern for batch normalization:
- Parameters are loaded once per channel (outer loop)
- Multiple spatial tiles per channel reuse the same parameters (inner loop)
- The `tile_start`/`counter` handles the case where a core's tile range starts mid-channel

This pattern is directly relevant to layer_norm_rm where a similar broadcast mechanism would be needed, but along the W dimension instead of the channel dimension.

### Conditional CB Routing

The `cb_affine_or_out` / `cb_scaled_output` mechanism avoids allocating separate CBs for each case. When weight and bias are both present, c_8 (temp_1) acts as a two-stage intermediate:
1. Normalized value -> c_8
2. Weighted value -> c_8 (overwriting previous)
3. Biased value -> c_2 (final output)

This works because each stage completes (push_back + pop_front) before the next stage writes to the same CB.

### `_with_dt` Helpers Pattern

All the `*_with_dt` functions from `moreh_common.hpp` follow the same pattern:
```cpp
ALWI void operation_with_dt(uint32_t icb0, uint32_t icb1) {
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(icb0, icb1);  // reconfigure unpack/math data format
#endif
    operation(icb0, icb1);  // call the base operation
}
```

These are essential when mixing CBs with different data formats (e.g., input in bfloat16, epsilon in bfloat16 but accumulating in fp32). The `reconfig_data_format` call ensures the hardware correctly interprets the CB data.

## External Knowledge Sources

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding `_with_dt` helper variants and composite compute operations
   **Key Information**: `_with_dt` wrappers add `reconfig_data_format` / `pack_reconfig_data_format` calls conditionally on `FP32_DEST_ACC_EN`. Provides `mul_tiles_to_cb`, `add_tiles_to_cb`, `sub_tiles_to_cb`, `copy_tile_to_cb` composite helpers that manage full CB lifecycle (reserve, wait, acquire, compute, commit, pack, release, pop, push).

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary.h`
   **Reason**: Understanding FPU binary operations and the `binary_dest_reuse_tiles` pattern
   **Key Information**: `binary_dest_reuse_tiles` with `DEST_TO_SRCA` moves DST[idst] to SRCA, reads from CB to SRCB, performs the binary op, writes result back to DST[idst]. This enables chaining without intermediate CB round-trips. Only unpacks from one CB (uses `llk_unpack_A`, not `llk_unpack_AB`).

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Understanding SFPU binary operations used in the fp32 path
   **Key Information**: SFPU binary ops (`add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`) operate entirely within DST registers. They take three DST indices (src0, src1, output). All use `llk_math_eltwise_binary_sfpu_binop` under the hood.

4. **Source**: `tt_metal/hw/inc/api/compute/tile_move_copy.h`
   **Reason**: Understanding `copy_tile_to_dst_init_short_with_dt`
   **Key Information**: Takes `old_cbid` and `new_cbid` parameters. Reconfigures data format for both unpack and math pipelines from old to new, then initializes copy. The old_cbid tells the reconfig what format to transition FROM.

5. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Understanding scalar broadcast mechanisms used for eps, mean, var, weight, bias
   **Key Information**: `fill_with_val_bfloat16(cb_id, packed_scalar)` fills 512 uint32 (1024 bfloat16 elements). `fill_tile_with_first_element_bfloat16(cb_id)` reads element[0], packs it, fills all 512 positions. Both ignore tile face structure since all elements are identical.

6. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Understanding core work distribution
   **Key Information**: `split_work_to_cores(grid_size, units_to_divide, row_major)` returns `{num_cores, all_cores, core_group_1, core_group_2, tiles_per_group_1, tiles_per_group_2}`. Group 1 gets ceil(work/cores), group 2 gets floor(work/cores).

7. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding CB creation helper
   **Key Information**: `create_cb(cb_index, program, core_spec, page_size, num_pages, data_format)` creates a circular buffer with `num_pages * page_size` total size. Returns `{cb_index, cb_handle}`.
