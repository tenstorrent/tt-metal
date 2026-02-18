# Batch Norm Implementation Analysis

## Overview

Batch normalization applies per-channel normalization to a 4D input tensor using pre-computed batch mean and batch variance statistics. The operation implements the formula:

```
output = (input - batch_mean) / sqrt(batch_var + eps) * weight + bias
```

where `weight` and `bias` are optional affine parameters. Critically, this operation does **not** compute the mean/variance itself -- those are provided as input tensors. The compute kernel performs only the **normalization** step (subtract mean, multiply by inverse standard deviation) and the optional affine transform.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Relevance to row_centralize**: This analysis focuses on the normalization compute pattern. For row_centralize, the compute kernel will need to perform mean reduction and variance computation *in addition to* the normalization step analyzed here. The batch norm compute kernel serves as the reference for the normalization-after-statistics phase.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `output.physical_volume() / tile_hw` = total output tiles |
| **Loop structure** | Outer loop over channels (freq = Ht * Wt tiles per channel), inner loop over spatial tiles within each channel. Statistics tensors (mean, var, weight, bias) are loaded once per channel and reused for all spatial tiles in that channel. |

The key insight is that batch norm statistics (mean, variance) are **per-channel** scalars, but the input has multiple spatial tiles per channel. The `freq` (frequency) parameter = `cHt * cWt` tells the compute kernel how many spatial tiles share the same channel statistics. Within each channel, one set of statistics tiles is loaded and broadcast across `freq` spatial tiles.

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | Batch Mean | Batch Var | Weight (optional) | Bias (optional) |
|----------|-----------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [N, C, 1, 1] or [1, C, 1, 1] | same as mean | same as mean | same as mean |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | matches input | matches input | matches input | matches input |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | Same as input [N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input (or specified via `dtype`) |

### Key Layout Details

The statistics tensors (mean, var, weight, bias) have shape `[N, C, 1, 1]` or `[1, C, 1, 1]`, meaning each is a single tile per channel (since 1x1 pads to one 32x32 tile). The writer kernel reads these tiles and calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the scalar value across the entire tile. This converts the per-channel scalar into a full tile that can be used in element-wise tile operations with the spatial input tiles.

## Data Flow Pattern

### High-Level Flow

```
DRAM[input]  --> Reader --> CB0 (input) --> Compute --> CB2 (output) --> Writer --> DRAM[output]
DRAM[mean]   --> Writer --> CB1 (mean)  --> Compute (consumed)
DRAM[var]    --> Writer --> CB3 (var)   --> Compute (consumed)
Reader fills CB4 (eps) once at startup --> Compute (consumed at end)
DRAM[weight] --> Writer --> CB5 (weight) --> Compute (consumed)
DRAM[bias]   --> Writer --> CB6 (bias)   --> Compute (consumed)
                           CB7 (den)    = intermediate (1/sqrt(var+eps))
                           CB8 (temp)   = intermediate (normalized result before affine)
```

### Critical Design: Split Reader Pattern

**The writer kernel also serves as a reader for statistics tensors.** This is a non-obvious but important design decision:

- **Reader kernel**: Reads only the input tensor (x) into CB0, and fills the epsilon CB (CB4) once.
- **Writer kernel**: Reads batch_mean (CB1), batch_var (CB3), weight (CB5), and bias (CB6) from DRAM, AND writes the output (CB2) to DRAM.

This split pattern has implications: the writer kernel runs on RISCV_1 / NOC1, which means all statistics reads and output writes go through NOC1. The reader kernel uses RISCV_0 / NOC0 exclusively for the main input tensor.

### Detailed Per-Channel Flow

For each channel (the loop frequency `freq = Ht * Wt`):

1. **Writer** reads 1 tile of batch_mean from DRAM, fills it with first element, pushes to CB1
2. **Writer** reads 1 tile of batch_var from DRAM, fills it with first element, pushes to CB3
3. **Writer** optionally reads weight tile, fills with first element, pushes to CB5
4. **Writer** optionally reads bias tile, fills with first element, pushes to CB6
5. **Compute** waits for CB3 (batch_var), adds eps from CB4, computes rsqrt, stores in CB7 (den)
6. For each spatial tile `t` in this channel:
   a. **Reader** reads input tile from DRAM, pushes to CB0
   b. **Compute** waits for CB0, subtracts CB1 (mean), multiplies by CB7 (den), produces result
   c. **Compute** optionally multiplies by CB5 (weight)
   d. **Compute** optionally adds CB6 (bias)
   e. **Compute** pushes final result to CB2
   f. **Writer** waits for CB2, writes tile to DRAM output
7. **Compute** pops mean, den, weight, bias from their CBs after all spatial tiles are processed

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | input_tensor_cb | Input data tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | batch_mean_tensor_cb | Per-channel mean (broadcast filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel |
| c_2 | output_tensor_cb | Normalized output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_3 | batch_var_tensor_cb | Per-channel variance (broadcast filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel |
| c_4 | eps_cb | Epsilon constant tile | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_5 | weight_tensor_cb | Optional affine weight (broadcast filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel |
| c_6 | bias_tensor_cb | Optional affine bias (broadcast filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel |
| c_7 | den_cb | 1/sqrt(batch_var + eps) intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Channel |
| c_8 | temp_1_cb | Intermediate for affine steps | 2 tiles | 1 tile | Double | Compute | Compute | Block |

### Notes on CB Lifetimes

- **Program lifetime (eps)**: The epsilon CB is filled once by the reader at startup, consumed by compute for every channel, and popped only at the very end.
- **Channel lifetime (mean, var, den, weight, bias)**: These are loaded once per channel, held in the CB while all spatial tiles in that channel are processed, then popped.
- **Block lifetime (input, output, temp)**: These cycle per tile -- one tile pushed, consumed, popped per iteration.

## Pipeline Pattern Summary

All CBs have capacity=2 and block_size=1, which is the **double-buffered** pattern. This allows the producer to start writing the next tile while the consumer processes the current one.

However, for the statistics CBs (c_1, c_3, c_5, c_6), the double-buffering is not used for pipelining overlap -- these tiles are held for the duration of the channel (popped only after all spatial tiles are processed). The extra capacity simply provides headroom.

## Index Calculations

### Tile Index to N/C/t Mapping

The program factory computes a flat `start_tile_id` per core. The kernels decompose this into batch, channel, and spatial-tile indices:

```cpp
uint32_t tiles_per_batch = HtWt * C;
uint32_t start_n = start_tile_id / tiles_per_batch;
uint32_t start_remaining = start_tile_id % tiles_per_batch;
uint32_t start_c = start_remaining / HtWt;
uint32_t start_t = start_remaining % HtWt;
```

This maps the linear tile ID to an (N, C, t) coordinate where `t` is the spatial position within a channel (ranging from 0 to Ht*Wt-1).

### Physical Tile Offset Calculation (Reader)

The reader computes a physical tile offset for the input tensor using stride multipliers:

```cpp
uint32_t tile_offset = start_n * n_stride + start_c * c_stride + start_t;
```

Where:
- `n_stride = aHt * aWt * aC * (aN > 1)`: stride between batches (0 if single batch)
- `c_stride = aHt * aWt * (aC > 1)`: stride between channels (0 if single channel)

The `(aN > 1)` and `(aC > 1)` guards handle degenerate dimensions. Within the inner loop, the tile_offset increments by 1 for each spatial tile, with jump corrections at channel and batch boundaries.

### Statistics Tile Offset (Writer)

For statistics tensors, the offset uses the batch_mean/var shape strides (bN, bC, bHt, bWt). Since these are `[N,C,1,1]` tensors, `bHt*bWt = 1`, so the offset increments by `c_stride` (which equals 1 for single-tile channels) at each channel boundary.

### Compute Frequency/Counter Mechanism

The compute kernel uses `freq` and `counter` to track channel boundaries:

```cpp
auto counter = start_tile_id % cHtWt;  // position within first channel
auto freq = cHtWt;                      // tiles per channel
```

- `freq = cHt * cWt`: number of spatial tiles sharing the same statistics
- `counter`: the starting offset within the first channel (handles partial channels at core boundaries)

The compute kernel calculates:
- `complete_iterations = (num_tiles + tile_start) / tile_freq`: number of full channel spans
- `remaining_iterations = (num_tiles + tile_start) % tile_freq`: leftover tiles in the last partial channel

Each call to `batchnorm_bcast_tiles()` processes one channel's worth of tiles (or a partial channel).

## Memory Access Patterns

### Read Pattern

**Input tensor (Reader)**: Sequential tile reads within each channel's spatial extent, with stride jumps at channel boundaries (`next_channel_shift = c_stride - HtWt`) and batch boundaries (`next_batch_shift = n_stride - c_stride * C`). Each tile read is a `noc_async_read_tile` with immediate barrier -- no batching of reads.

**Statistics tensors (Writer)**: One tile read per channel (per statistics tensor), accessed via tile_offset that increments by `c_stride` per channel. After reading, `FILL_TILE_WITH_FIRST_ELEMENT` broadcasts the scalar across the full tile in L1.

### Write Pattern

**Output tensor (Writer)**: Sequential tile writes using a flat `start_tile_id + num_tiles_written` index via `noc_async_write_tile`. One tile written per iteration, with immediate barrier.

### Access Characteristics

- All reads/writes are to DRAM (interleaved layout)
- TensorAccessor handles bank-to-physical-address mapping
- No L1 sharding
- Each NoC read/write is 1 tile at a time with a barrier after each -- no batch prefetching

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (using full compute grid) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | num_cores_x * num_cores_y |
| **Work per core** | `num_output_tiles / num_cores` tiles (with remainder handling) |
| **Load balancing** | Two-group split via `split_work_to_cores` |

### Work Distribution Details

`split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major=true)` divides output tiles across cores:

- **core_group_1**: Gets `ceil(num_output_tiles / num_cores)` tiles each
- **core_group_2**: Gets `floor(num_output_tiles / num_cores)` tiles each
- Cores beyond what is needed get 0 tiles and early-exit (runtime args all set to 0)

The `start_tile_id` accumulates across cores: each core picks up where the previous one left off in the flat tile index space.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src | uint32_t | CB index for input tensor (c_0) |
| 1 | cb_id_eps | uint32_t | CB index for epsilon tile (c_4) |
| 2+ | TensorAccessorArgs(input) | uint32_t[] | Accessor args for input tensor (bank mapping, shapes) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | Boolean: weight tensor present |
| 1 | bias_has_value | uint32_t | Boolean: bias tensor present |
| 2 | cb_id_src (mean) | uint32_t | CB index for batch_mean (c_1) |
| 3 | cb_id_dst (output) | uint32_t | CB index for output (c_2) |
| 4 | cb_id_batch_var | uint32_t | CB index for batch_var (c_3) |
| 5 | cb_id_weight | uint32_t | CB index for weight (c_5) |
| 6 | cb_id_bias | uint32_t | CB index for bias (c_6) |
| 7+ | TensorAccessorArgs(batch_mean) | uint32_t[] | Accessor args for mean tensor |
| ... | TensorAccessorArgs(output) | uint32_t[] | Accessor args for output tensor |
| ... | TensorAccessorArgs(batch_var) | uint32_t[] | Accessor args for var tensor |
| ... | TensorAccessorArgs(weight) | uint32_t[] | Accessor args for weight tensor |
| ... | TensorAccessorArgs(bias) | uint32_t[] | Accessor args for bias tensor |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | Boolean: weight tensor present |
| 1 | bias_has_value | uint32_t | Boolean: bias tensor present |
| 2 | cb_input | uint32_t | CB index for input (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon (c_4) |
| 7 | cb_den | uint32_t | CB index for denominator intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB index for weight (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for temp intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB index for bias (c_6) |

### Runtime Arguments

#### Reader Kernel (11 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value packed as bf16 pair or float32 bits |
| 1 | src_addr | uint32_t | Input tensor buffer address in DRAM |
| 2 | start_tile_id | uint32_t | First output tile index for this core |
| 3 | num_tiles | uint32_t | Total tiles to process on this core |
| 4 | HtWt | uint32_t | cHt * cWt = spatial tiles per channel |
| 5 | n_stride | uint32_t | Input tile stride between batches |
| 6 | c_stride | uint32_t | Input tile stride between channels |
| 7 | N | uint32_t | Number of output batches |
| 8 | C | uint32_t | Number of output channels |
| 9 | Ht | uint32_t | Output height in tiles (unused in reader -- leftover from shape) |
| 10 | Wt | uint32_t | Output width in tiles (unused in reader -- leftover from shape) |

#### Writer Kernel (14 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Batch mean tensor buffer address |
| 1 | batch_var_addr | uint32_t | Batch variance tensor buffer address |
| 2 | weight_addr | uint32_t | Weight tensor buffer address (0 if absent) |
| 3 | bias_addr | uint32_t | Bias tensor buffer address (0 if absent) |
| 4 | dst_addr | uint32_t | Output tensor buffer address |
| 5 | start_tile_id | uint32_t | First output tile index for this core |
| 6 | num_tiles | uint32_t | Total tiles to process on this core |
| 7 | HtWt | uint32_t | Spatial tiles per channel |
| 8 | n_stride | uint32_t | Statistics tile stride between batches |
| 9 | c_stride | uint32_t | Statistics tile stride between channels |
| 10 | N | uint32_t | Number of output batches |
| 11 | C | uint32_t | Number of output channels |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |

#### Compute Kernel (3 args)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | tile_freq | uint32_t | HtWt = tiles per channel (frequency of statistics reload) |
| 2 | tile_start | uint32_t | `start_tile_id % cHtWt` = offset into first channel |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM[input] | CB0 (input), CB4 (eps) | Read input tiles, fill eps tile once |

- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp`
- **Key Logic**:
  - At startup, fills CB4 with the epsilon value using `fill_with_val_bfloat16` (bf16 path) or `fill_with_val<1024, float>` (fp32 path). The epsilon is packed as `(bf16 << 16 | bf16)` for bf16 or as raw float32 bits.
  - Iterates in N -> C -> t order, reading one input tile at a time.
  - Uses `noc_async_read_tile(tile_offset, src, l1_write_addr)` with TensorAccessor for address resolution.
  - Each read has an immediate barrier (`noc_async_read_barrier()`) -- no prefetching.
  - Handles batch/channel stride jumps via `next_channel_shift` and `next_batch_shift`.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_batch_norm | RISCV_1 | NOC1 | DRAM[mean, var, weight, bias], CB2 (output) | CB1 (mean), CB3 (var), CB5 (weight), CB6 (bias), DRAM[output] | Read statistics, broadcast-fill tiles, write output |

- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`
- **Key Logic**:
  - Iterates in N -> C -> t order, matching the reader/compute loop structure.
  - Per channel: reads one tile each of mean, var, (optionally weight, bias) from DRAM. Each tile is broadcast-filled using `FILL_TILE_WITH_FIRST_ELEMENT` so the first element (the per-channel scalar) fills the entire 32x32 tile.
  - Per spatial tile within the channel: waits for the compute output in CB2, writes it to DRAM using `noc_async_write_tile`.
  - The tile_offset for statistics tensors uses the batch_mean/var shape strides, advancing by `c_stride` per channel.

### Compute Kernel (FPU variant: `batch_norm_kernel.cpp`)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| compute_batch_norm | RISCV_2 (FPU) | N/A | CB0, CB1, CB3, CB4, CB5, CB6 | CB2 (output), CB7 (den), CB8 (temp) | sub, mul, add, rsqrt (via FPU binary ops) |

- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- **Key Logic** (the normalization compute pattern):

**Per-channel preamble (run once per channel):**
1. Wait for batch_var (CB3), compute `var + eps` using `add_tiles`, then `rsqrt_tile` to get `1/sqrt(var + eps)`, pack to CB7 (den).
2. Wait for mean (CB1), den (CB7), and optionally weight (CB5), bias (CB6).

**Per spatial tile (run `freq` times per channel):**
3. Wait for input tile (CB0).
4. `sub_tiles(input, mean)` -- subtract mean, result stays in DST register.
5. `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(den)` -- multiply by 1/sqrt(var+eps), **reusing the subtraction result already in DST as SRCA**. This avoids writing the intermediate subtraction result to a CB.
6. Pack result to `cb_affine_or_out` (either CB8 or CB2 depending on whether affine transform is needed).
7. Pop input from CB0.

**Optional weight multiplication:**
8. Wait for intermediate (CB8), mul_tiles with weight (CB5), pack to `cb_scaled_output`.

**Optional bias addition:**
9. Wait for intermediate (CB8), add_tiles with bias (CB6), pack to CB2 (output).

**End of channel:**
10. Pop mean (CB1), den (CB7), and optionally weight (CB5), bias (CB6).

**The `binary_dest_reuse_tiles` optimization** is the key compute pattern: the subtraction `(input - mean)` leaves the result in DST register slot 0. Instead of packing to a CB and unpacking, `DEST_TO_SRCA` reuses DST as the source A operand for the subsequent multiplication by `den`. This eliminates one CB write/read cycle per tile.

### Compute Kernel (SFPU variant: `batch_norm_sfpu_kernel.cpp`)

- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`
- **Selected when**: `fp32_dest_acc_en` is true (FP32 destination accumulation mode)
- **Key Differences from FPU variant**:
  - Uses `eltwise_binary_sfpu.h` instead of `eltwise_binary.h`
  - Uses `copy_tile_to_dst` + `add_binary_tile` / `sub_binary_tile` / `mul_binary_tile` instead of `add_tiles` / `sub_tiles` / `mul_tiles`
  - SFPU binary ops work by copying operands to DST register slots (slot 0 and slot 1), then performing the operation. Example: `copy_tile(cb_batch_var, i, i*2)` then `copy_tile(cb_eps, i, i*2+1)` then `add_binary_tile(i*2, i*2+1, i*2)`.
  - Uses `unary_op_init_common` instead of `binary_op_init_common`.
  - Same mathematical logic, different hardware execution path (vector engine vs matrix engine).

### Compute Kernel Selection Logic

From the program factory (line 288-289):
```cpp
fmt::format("...batch_norm_{}.cpp", fp32_dest_acc_en ? "sfpu_kernel" : "kernel")
```

When FP32 destination accumulation is enabled, the SFPU variant is selected. The SFPU variant also sets `UnpackToDestMode::UnpackToDestFp32` for all input CBs to enable FP32 precision in the destination register.

## Implementation Notes

### Epsilon Packing Convention

The reader packs epsilon differently based on data type:
- **FLOAT32**: `std::bit_cast<uint32_t>(eps)` -- raw float32 bits
- **BFLOAT16**: `pack_two_bfloat16_into_uint32({eps, eps})` -- bf16 value duplicated into both halves

The reader then uses `fill_with_val<1024, float>` (fp32) or `fill_with_val_bfloat16` (bf16) to fill the entire tile with this value.

### FILL_TILE_WITH_FIRST_ELEMENT Pattern

This is a critical dataflow pattern for broadcasting per-channel scalars. The statistics tensors have shape [N,C,1,1], so each tile contains a single meaningful value in the first element. `fill_tile_with_first_element_bfloat16` reads the first uint16 element, packs it into both halves of a uint32, then writes it across all 512 uint32 positions (1024 bf16 elements) to fill the entire 32x32 tile. This converts a scalar into a full tile that can be used in element-wise tile operations.

### CB Routing for Affine Transform

The compute kernel dynamically routes output to different CBs depending on which optional parameters are present:

```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```

- **No weight, no bias**: normalized result goes directly to output CB (c_2)
- **Weight but no bias**: normalized result goes to temp CB (c_8), weight multiply goes to output CB (c_2)
- **Both weight and bias**: normalized result goes to temp CB (c_8), weight multiply goes to temp CB (c_8), bias add goes to output CB (c_2)
- **Bias but no weight**: normalized result goes to temp CB (c_8), bias add goes to output CB (c_2)

### Defines-Based Dataflow Configuration

The program factory defines macros (`FILL_TILE_WITH_FIRST_ELEMENT`, `FILL_WITH_VALUE`, `FILL_WITH_VALUE_FLOAT`) based on input dtype. These macros are passed to the reader and writer kernels via `dataflow_defines`, enabling the same kernel source to handle both bf16 and fp32 data types without runtime branching.

### Program Caching

The operation supports program caching via `override_runtime_arguments`, which updates runtime args for all cores without recreating kernels. The `shared_variables_t` struct stores kernel handles and grid size for reuse.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What are binary_dest_reuse_tiles and binary_dest_reuse_tiles_init in tt-metal compute kernels? How does DEST_TO_SRCA work for chaining operations without intermediate CB writes?"
   **Reason**: The FPU compute kernel uses `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` which is the key optimization for chaining the subtraction and multiplication without an intermediate CB write.
   **Key Findings**: `DEST_TO_SRCA` loads the current DST register value into Source A, then performs the binary operation using Source A (previous result) and Source B (new operand from CB). This avoids writing the intermediate result to a CB and reading it back, keeping data in fast DST registers. The result stays in DST for potential further chaining.

2. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: The program factory uses this to distribute output tiles across cores.
   **Key Findings**: Returns (num_cores, all_cores, core_group_1, core_group_2, tiles_per_group_1, tiles_per_group_2). When work is not evenly divisible, core_group_1 gets one extra tile each. The number of cores in group_1 equals `units_to_divide % target_num_cores`.

3. **Query**: "What is the SFPU compute kernel variant versus the FPU variant?"
   **Reason**: Batch norm has two compute kernel variants selected by `fp32_dest_acc_en`.
   **Key Findings**: FPU (matrix engine) handles standard add/sub/mul via `eltwise_binary.h`. SFPU (vector engine) handles the same via `eltwise_binary_sfpu.h` with explicit `copy_tile_to_dst` + binary_tile pattern. SFPU is used when FP32 destination accumulation is needed for higher precision.

### Documentation References

1. **Source**: `tech_reports/tensor_accessor/tensor_accessor.md`
   **Reason**: Understanding how TensorAccessor maps tile indices to DRAM addresses in the reader/writer kernels.
   **Key Information**: TensorAccessorArgs are passed as compile-time args, TensorAccessor is constructed device-side with `TensorAccessor(args, base_addr, tile_size)`. `noc_async_read_tile(tile_id, accessor, l1_addr)` resolves the tile to its physical bank and offset.

2. **Source**: `tt_metal/api/tt-metalium/work_split.hpp`
   **Reason**: Understanding the two-group work distribution pattern.
   **Key Information**: `split_work_to_cores` returns 6-tuple: num_cores, all_cores, core_group_1, core_group_2, tiles_per_group_1, tiles_per_group_2. Handles remainder by giving some cores one extra tile.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Understanding the broadcast-fill mechanism for statistics tiles.
   **Key Information**: `fill_tile_with_first_element_bfloat16` reads first element, packs it into both halves of uint32, writes across all 512 positions. `fill_with_val_bfloat16` fills with a provided packed scalar. These convert scalars to full tiles for element-wise operations.

4. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding `pack_tile_with_dt`, `add_tiles_init_with_dt`, etc.
   **Key Information**: The `_with_dt` variants add `reconfig_data_format` calls when FP32_DEST_ACC_EN is defined, ensuring correct data format configuration when mixing data types. These are wrappers around the standard compute APIs.
