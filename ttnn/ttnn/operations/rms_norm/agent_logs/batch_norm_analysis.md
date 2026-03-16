# Batch Norm Implementation Analysis (Compute-Core Focus)

## Overview

Batch normalization computes: `output = ((input - batch_mean) / sqrt(batch_var + eps)) * weight + bias`

The batch_mean and batch_var are pre-computed and passed as separate 1D tensors (per-channel scalars). The operation broadcasts these per-channel values across the spatial (H, W) tiles of each channel.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

This analysis focuses on **compute kernel structure** as a reference for implementing RMSNorm. Reader/writer details are noted only for what they provide to and consume from the compute kernel.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `output.physical_volume() / tile_hw` |
| **Loop structure** | Two-level: outer loop iterates over "channel groups" (complete_iterations + remainder), inner loop iterates over tiles within a channel group (tile_start..freq). Each channel group recomputes `1/sqrt(var+eps)` and reloads per-channel params, then processes `freq` spatial tiles using those broadcast values. |

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | batch_mean | batch_var | weight (optional) | bias (optional) |
|----------|-----------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | same | same | same | same |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [N, C, H, W] (same as input) |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Data type** | same as input |

### Important Note: Per-channel Broadcast Pattern

The per-channel tensors (batch_mean, batch_var, weight, bias) are 1D-like (one scalar per channel). The writer kernel reads one tile from each per-channel tensor per channel, then calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the first element of that tile across all 1024 positions. This creates a full tile where every element equals the channel's scalar value, enabling standard tile-level elementwise operations in the compute kernel.

## Data Flow Pattern

### What Reader Provides

1. **Epsilon CB (c_4)**: Fills one tile with the epsilon scalar value using `fill_with_val` or `fill_with_val_bfloat16` at startup. This tile persists for the entire kernel lifetime.
2. **Input CB (c_0)**: Reads input tiles one-at-a-time in N->C->HtWt order. One tile per push.

### What Writer Provides

For each channel (not each tile), the writer reads and broadcasts:
1. **batch_mean CB (c_1)**: One broadcast-filled tile per channel.
2. **batch_var CB (c_3)**: One broadcast-filled tile per channel.
3. **weight CB (c_5)**: One broadcast-filled tile per channel (if weight exists).
4. **bias CB (c_6)**: One broadcast-filled tile per channel (if bias exists).

Then, for each spatial tile within that channel, the writer consumes one output tile from **output CB (c_2)**.

### What Compute Produces

For each channel group:
1. Computes `1/sqrt(batch_var + eps)` into **den CB (c_7)** -- one tile, reused across spatial tiles.
2. For each spatial tile: computes `(input - batch_mean) * den`, optionally `* weight`, optionally `+ bias`, producing one tile to **output CB (c_2)**.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | input_tensor_cb | Input tiles | 2 tiles | 1 tile | Double | Reader | Compute | Block (per spatial tile) |
| c_1 | batch_mean_tensor_cb | Per-channel mean (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel group) |
| c_2 | output_tensor_cb | Output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block (per spatial tile) |
| c_3 | batch_var_tensor_cb | Per-channel variance (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Block (consumed per channel group) |
| c_4 | eps_cb | Epsilon scalar tile | 2 tiles | 1 tile | Double | Reader | Compute | Program (entire kernel) |
| c_5 | weight_tensor_cb | Per-channel weight (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel group) |
| c_6 | bias_tensor_cb | Per-channel bias (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Row (per channel group) |
| c_7 | den_cb | `1/sqrt(batch_var + eps)` intermediate | 2 tiles | 1 tile | Double | Compute | Compute | Row (per channel group) |
| c_8 | temp_1_cb | Intermediate for affine transform | 2 tiles | 1 tile | Double | Compute | Compute | Block (per spatial tile) |

### Multi-Pass Data Reuse Patterns (Critical for RMSNorm Design)

The batch_norm compute kernel demonstrates a key architectural pattern: **per-channel values persist across multiple spatial tiles within a channel group**. Specifically:

1. **cb_eps (c_4)**: Loaded once by reader at startup. `wait_front(1)` at start of `kernel_main()`, `pop_front(1)` at the very end. **Persists for the entire program**.

2. **cb_bcast / cb_batch_mean (c_1)**: Writer pushes 1 tile per channel. Compute does `wait_front(1)` at the top of `batchnorm_bcast_tiles()`, then reads it repeatedly in the inner loop for each spatial tile. `pop_front(1)` only happens **after all spatial tiles in that channel group are processed**.

3. **cb_den (c_7)**: Compute-internal. Produced once per channel group from `batch_var + eps`, then consumed repeatedly in the inner spatial loop. Popped after the channel group completes.

4. **cb_weight (c_5) / cb_bias (c_6)**: Same as cb_bcast -- wait once, reuse across spatial tiles, pop at the end of the channel group.

5. **cb_input (c_0), cb_output (c_2), cb_tmp_1 (c_8)**: Transient -- produced and consumed once per spatial tile iteration. These are single-tile-at-a-time streaming.

**Implication for RMSNorm**: RMSNorm needs a different reuse pattern. Instead of "per-channel scalar persists across spatial tiles", RMSNorm needs "input tiles persist across two passes" (first pass: compute mean(x^2), second pass: multiply x by 1/sqrt(mean+eps)). This is because RMSNorm's reduction is over the last dimension (width tiles), requiring all tiles in a row to be accumulated before the scalar result can be applied.

## Compute Kernel Structure (Two Variants)

### Variant Selection

The program factory selects between two compute kernel variants at lines 285-293:
- `batch_norm_kernel.cpp` -- FPU path (when `fp32_dest_acc_en == false`)
- `batch_norm_sfpu_kernel.cpp` -- SFPU path (when `fp32_dest_acc_en == true`)

Both have identical structure and mathematical logic. The difference is in which hardware unit performs the computation and how tile registers are managed.

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if weight tensor provided, 0 otherwise |
| 1 | bias_has_value | uint32_t | 1 if bias tensor provided, 0 otherwise |
| 2 | cb_input | uint32_t | CB ID for input (c_0) |
| 3 | cb_batch_mean | uint32_t | CB ID for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB ID for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB ID for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB ID for epsilon tile (c_4) |
| 7 | cb_den | uint32_t | CB ID for 1/sqrt(var+eps) intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB ID for weight (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB ID for intermediate (c_8) |
| 10 | cb_bias | uint32_t | CB ID for bias (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total number of output tiles assigned to this core |
| 1 | tile_freq | uint32_t | Number of spatial tiles per channel group (`cHt * cWt`) |
| 2 | tile_start | uint32_t | Starting offset within first channel group (`start_tile_id % cHtWt`) |

### Channel Group Iteration Logic

The `tile_freq` and `tile_start` arguments enable correct processing when a core's tile assignment spans partial channel groups:

```
complete_iterations = (num_tiles + tile_start) / tile_freq
remaining_iterations = (num_tiles + tile_start) % tile_freq
```

- First call to `batchnorm_bcast_tiles()`: processes from `tile_start` to `tile_freq` (partial first group).
- Subsequent calls: process full channel groups (0 to `tile_freq`).
- Final call (if `remaining_iterations > 0`): processes partial last group.

After the first iteration, `tile_start` is reset to 0 (`tile_start = 0` in the for loop increment).

## Compute Kernel Detailed Logic (FPU Variant: `batch_norm_kernel.cpp`)

### Initialization

```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```

This initializes the FPU binary operation hardware for the input/broadcast/output CB combination.

### Epsilon CB Handling

```cpp
cb_eps_obj.wait_front(onetile);  // At start -- persists entire program
// ... all processing ...
cb_eps_obj.pop_front(onetile);   // At end of kernel_main
```

### `batchnorm_bcast_tiles()` Function -- Phase-by-Phase

#### Phase 1: Compute Denominator (Once Per Channel Group)

```cpp
// Compute: cb_den = 1/sqrt(batch_var + eps)
cb_den_obj.reserve_back(onetile);
cb_batch_var_obj.wait_front(onetile);

tile_regs_acquire();
add_tiles_init_with_dt(cb_batch_var, cb_eps);
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);   // dst0 = batch_var + eps
rsqrt_tile_init();
rsqrt_tile(dst0);                                // dst0 = 1/sqrt(batch_var + eps)
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(dst0, cb_den);                 // pack to cb_den
tile_regs_release();

cb_batch_var_obj.pop_front(onetile);
cb_den_obj.push_back(onetile);
```

**Key API calls:**
- `add_tiles_init_with_dt(icb0, icb1)`: Initializes the FPU add operation with data type reconfiguration (calls `reconfig_data_format(icb0, icb1)` when FP32_DEST_ACC_EN).
- `add_tiles(icb0, icb1, itile0, itile1, dst)`: FPU elementwise addition: `dst = icb0[itile0] + icb1[itile1]`.
- `rsqrt_tile_init()`: Initializes the SFPU for reciprocal square root.
- `rsqrt_tile(dst)`: In-place SFPU operation: `dst = 1/sqrt(dst)`.
- `pack_tile_with_dt(dst, ocb)`: Packs tile from dest register to CB, with data format reconfiguration if FP32_DEST_ACC_EN.

**Important**: `rsqrt_tile` is an SFPU operation even in the FPU kernel variant. The FPU kernel uses FPU for binary ops (add, sub, mul) but switches to SFPU for transcendental functions like rsqrt.

#### Phase 2: Wait for Broadcast Tiles (Once Per Channel Group)

```cpp
cb_bcast_obj.wait_front(onetile);    // batch_mean -- stays for entire channel group
cb_den_obj.wait_front(onetile);      // 1/sqrt(var+eps) -- stays for entire channel group
if (weight_has_value) cb_weight_obj.wait_front(onetile);
if (bias_has_value) cb_bias_obj.wait_front(onetile);
```

These tiles are NOT popped inside the inner loop. They persist until all spatial tiles in the channel group are processed.

#### Phase 3: Inner Loop (Per Spatial Tile)

**CB routing logic** (determines which CB the intermediate goes to):
```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```

This routing avoids unnecessary CB hops. If no weight/bias, the normalized result goes directly to output.

**Step 3a: Subtract mean and multiply by denominator**

```cpp
cb_other_obj.wait_front(onetile);           // wait for input tile
cb_affine_or_out_obj.reserve_back(onetile);

tile_regs_acquire();
sub_tiles_init(cb_other, cb_bcast);
sub_tiles(cb_other, cb_bcast, 0, 0, 0);    // dst0 = input - batch_mean

// Reuse dst0 as SrcA, multiply with den from CB
binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
// dst0 = (input - batch_mean) * 1/sqrt(var+eps)
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(0, cb_affine_or_out);
tile_regs_release();

cb_affine_or_out_obj.push_back(onetile);
cb_other_obj.pop_front(onetile);
```

**Key pattern: `binary_dest_reuse_tiles`**

This is a critical optimization. Instead of packing the subtraction result to a CB and then reading it back for multiplication, `binary_dest_reuse_tiles` keeps the result in the dest register and uses it directly as SrcA for the next operation:

- `DEST_TO_SRCA`: The value already in dst0 (from `sub_tiles`) is moved to SrcA.
- The CB tile (`cb_den`, tile 0) is loaded into SrcB.
- The multiply is performed: `dst0 = SrcA * SrcB = (input - mean) * den`.

This saves one pack+unpack round trip and the corresponding CB space.

**Step 3b: Multiply by weight (conditional)**

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

Note: `cb_weight` tile index is always 0 because the same weight tile is reused for all spatial tiles in the channel group (it was not popped).

**Step 3c: Add bias (conditional)**

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

#### Phase 4: Cleanup (After Inner Loop)

```cpp
cb_bcast_obj.pop_front(onetile);     // release batch_mean
cb_den_obj.pop_front(onetile);       // release denominator
if (weight_has_value) cb_weight_obj.pop_front(onetile);
if (bias_has_value) cb_bias_obj.pop_front(onetile);
```

## SFPU Variant Differences (`batch_norm_sfpu_kernel.cpp`)

The SFPU variant uses explicit `copy_tile` to move data into dest registers, followed by SFPU binary operations:

### Initialization
```cpp
unary_op_init_common(cb_other, cb_output_0);  // vs binary_op_init_common in FPU variant
```

### Phase 1 (Denominator) -- SFPU Pattern

```cpp
tile_regs_acquire();
tile_regs_wait();

// Load batch_var into dst[0]
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
copy_tile(cb_batch_var, 0, 0);  // dst[0] = batch_var

// Load eps into dst[1]
add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
copy_tile(cb_eps, 0, 1);        // dst[1] = eps

add_binary_tile(0, 1, 0);       // dst[0] = dst[0] + dst[1] = batch_var + eps

rsqrt_tile_init();
rsqrt_tile(0);                  // dst[0] = 1/sqrt(batch_var + eps)

pack_tile(0, cb_den);           // note: pack_tile not pack_tile_with_dt
tile_regs_commit();
tile_regs_release();
```

**Key SFPU API differences:**
- `copy_tile_to_dst_init_short_with_dt(old_cb, new_cb)`: Reconfigures unpack for the new CB's data format.
- `copy_tile(cb, tile_idx, dst_idx)`: Unpacks tile from CB into dest register at `dst_idx`.
- `add_binary_tile(dst_a, dst_b, dst_out)`: SFPU add on dest registers. `dst_out = dst_a + dst_b`.
- `sub_binary_tile(dst_a, dst_b, dst_out)`: SFPU subtract. `dst_out = dst_a - dst_b`.
- `mul_binary_tile(dst_a, dst_b, dst_out)`: SFPU multiply. `dst_out = dst_a * dst_b`.
- `pack_tile(dst_idx, cb)`: Pack without data type reconfiguration (vs `pack_tile_with_dt`).

**SFPU uses even/odd dst indexing**: Operations use `i*2` and `i*2+1` for the two operands, with the result going to `i*2`. This is because the SFPU binary operations work on two distinct dest register slots.

## FPU vs SFPU Path Summary

| Aspect | FPU Path (`batch_norm_kernel.cpp`) | SFPU Path (`batch_norm_sfpu_kernel.cpp`) |
|--------|-------------------------------------|------------------------------------------|
| **When selected** | `fp32_dest_acc_en == false` | `fp32_dest_acc_en == true` |
| **Init** | `binary_op_init_common()` | `unary_op_init_common()` |
| **Binary ops** | `add_tiles()`, `sub_tiles()`, `mul_tiles()` | `add_binary_tile()`, `sub_binary_tile()`, `mul_binary_tile()` |
| **Operand loading** | Implicit (from CB directly) | Explicit `copy_tile()` into dest |
| **Data format reconfig** | `_with_dt` helpers | `copy_tile_to_dst_init_short_with_dt` |
| **Pack** | `pack_tile_with_dt()` | `pack_tile()` |
| **Dest reuse optimization** | `binary_dest_reuse_tiles` | Manual dest register indexing |
| **Dest register usage** | Typically dst[0] only | Uses pairs dst[i*2] and dst[i*2+1] |

## Pipeline Pattern Summary

| CB | Capacity | Block Size | Buffering Type |
|----|----------|------------|----------------|
| c_0 (input) | 2 tiles | 1 tile | Double-buffered |
| c_1 (batch_mean) | 2 tiles | 1 tile | Double-buffered |
| c_2 (output) | 2 tiles | 1 tile | Double-buffered |
| c_3 (batch_var) | 2 tiles | 1 tile | Double-buffered |
| c_4 (eps) | 2 tiles | 1 tile | Double-buffered |
| c_5 (weight) | 2 tiles | 1 tile | Double-buffered |
| c_6 (bias) | 2 tiles | 1 tile | Double-buffered |
| c_7 (den) | 2 tiles | 1 tile | Double-buffered |
| c_8 (temp_1) | 2 tiles | 1 tile | Double-buffered |

All CBs use capacity=2 (double-buffered), though the long-lived broadcast tiles (batch_mean, weight, bias, den) only occupy 1 slot persistently.

## Index Calculations

### Compute Kernel Indexing

The compute kernel does not perform complex index calculations. It receives:
- `num_tiles`: total tiles to process
- `tile_freq` (`cHt * cWt`): spatial tiles per channel
- `tile_start`: starting offset within first channel group

The iteration pattern is:
```
for each complete channel group:
    load per-channel params (mean, var, weight, bias)
    for each spatial tile in group:
        process one tile
if remaining tiles:
    process partial channel group
```

### Host-Side Tile ID to Channel Mapping (Program Factory)

```cpp
uint32_t cHtWt = cHt * cWt;
auto counter = start_tile_id % cHtWt;  // = tile_start
auto freq = cHtWt;                      // = tile_freq
```

The reader/writer map the flat `start_tile_id` to (N, C, HtWt) coordinates using:
```
tiles_per_batch = HtWt * C
start_n = start_tile_id / tiles_per_batch
start_remaining = start_tile_id % tiles_per_batch
start_c = start_remaining / HtWt
start_t = start_remaining % HtWt
```

## Memory Access Patterns

### Read Pattern (Compute perspective)
- **Input tiles (c_0)**: Sequential, one tile at a time, streaming (read once, pop immediately after use in subtraction).
- **Broadcast tiles (c_1, c_3, c_5, c_6)**: Read once per channel group, held in CB for multiple iterations, then popped.
- **Epsilon (c_4)**: Read once at program start, held for entire execution.

### Write Pattern (Compute perspective)
- **Output tiles (c_2)**: Sequential, one tile at a time, immediately available for writer after push.
- **Intermediate tiles (c_7, c_8)**: Compute-internal, short-lived within the channel group processing.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (flattened row-major) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores_x * num_cores_y` (device-dependent) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` output tiles |
| **Load balancing** | Two-group split via `split_work_to_cores()` -- group1 gets `ceil(total/cores)` tiles, group2 gets `floor(total/cores)` |
| **Remainder handling** | Cores not in group1 or group2 receive zero-initialized args and return immediately |

## Arguments Summary

### Reader Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src (input_tensor_cb) | uint32_t | CB ID for input (c_0) |
| 1 | cb_id_eps (eps_cb) | uint32_t | CB ID for epsilon (c_4) |
| 2+ | TensorAccessorArgs | uint32_t[] | Input tensor accessor parameters |

### Writer Compile-Time Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | Whether weight tensor is present |
| 1 | bias_has_value | uint32_t | Whether bias tensor is present |
| 2 | batch_mean_tensor_cb | uint32_t | CB ID for batch mean (c_1) |
| 3 | output_tensor_cb | uint32_t | CB ID for output (c_2) |
| 4 | batch_var_tensor_cb | uint32_t | CB ID for batch var (c_3) |
| 5 | weight_tensor_cb | uint32_t | CB ID for weight (c_5) |
| 6 | bias_tensor_cb | uint32_t | CB ID for bias (c_6) |
| 7+ | TensorAccessorArgs | uint32_t[] | For batch_mean, output, batch_var, weight, bias |

### Reader Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value packed as float32 or two bfloat16s |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | First output tile assigned to this core |
| 3 | num_tiles | uint32_t | Number of tiles to process |
| 4 | HtWt | uint32_t | Spatial tiles per channel (cHt * cWt) |
| 5 | n_stride | uint32_t | Tile stride for batch dimension |
| 6 | c_stride | uint32_t | Tile stride for channel dimension |
| 7 | N | uint32_t | Batch count of output |
| 8 | C | uint32_t | Channel count of output |
| 9 | Ht | uint32_t | Height in tiles of output |
| 10 | Wt | uint32_t | Width in tiles of output |

### Writer Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | batch_mean buffer address |
| 1 | batch_var_addr | uint32_t | batch_var buffer address |
| 2 | weight_addr | uint32_t | weight buffer address (0 if absent) |
| 3 | bias_addr | uint32_t | bias buffer address (0 if absent) |
| 4 | dst_addr | uint32_t | output buffer address |
| 5 | start_tile_id | uint32_t | First output tile assigned to this core |
| 6 | num_tiles | uint32_t | Number of tiles to process |
| 7 | HtWt | uint32_t | Spatial tiles per channel |
| 8 | n_stride | uint32_t | Tile stride for batch dimension |
| 9 | c_stride | uint32_t | Tile stride for channel dimension |
| 10 | N | uint32_t | Batch count |
| 11 | C | uint32_t | Channel count |
| 12 | Ht | uint32_t | Height in tiles |
| 13 | Wt | uint32_t | Width in tiles |

### Compute Runtime Arguments

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total output tiles to process on this core |
| 1 | tile_freq | uint32_t | Spatial tiles per channel group (cHt * cWt) |
| 2 | tile_start | uint32_t | Offset within first channel group |

## Kernel Implementations

| Kernel | Core | NOC | Input CBs | Output CBs | Operations |
|--------|------|-----|-----------|------------|------------|
| reader | RISCV_0 | NOC0 | DRAM (input) | c_0 (input), c_4 (eps) | Read input tiles; fill eps tile once |
| writer | RISCV_1 | NOC1 | DRAM (mean,var,weight,bias) | c_1,c_3,c_5,c_6; consumes c_2 | Read+broadcast per-channel params; write output |
| compute (FPU) | RISCV_2 | N/A | c_0,c_1,c_3,c_4,c_5,c_6 | c_2 (output), c_7,c_8 (internal) | sub, mul, rsqrt, add |
| compute (SFPU) | RISCV_2 | N/A | c_0,c_1,c_3,c_4,c_5,c_6 | c_2 (output), c_7,c_8 (internal) | copy_tile, binary_tile ops, rsqrt |

### Compute Kernel Files
- **FPU variant**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- **SFPU variant**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`

## Implementation Notes

### UnpackToDestMode Configuration

When `fp32_dest_acc_en` is true, all CBs involved in computation are configured with `UnpackToDestMode::UnpackToDestFp32` (lines 258-271 of program factory). This ensures data is unpacked to the dest register file in FP32 format for higher precision.

### Conditional CB Routing for Minimal Memory Traffic

The `cb_affine_or_out` and `cb_scaled_output` variables implement smart CB routing:
- No weight, no bias: normalized result goes directly to `cb_output_0`.
- Weight but no bias: normalized result to `cb_tmp_1`, then weight-scaled result to `cb_output_0`.
- Weight and bias: normalized result to `cb_tmp_1`, weight-scaled result stays in `cb_tmp_1`, then bias-added result to `cb_output_0`.

This avoids unnecessary intermediate CB round-trips when optional parameters are absent.

### Data Format Defines for Reader/Writer

The reader/writer kernels use `#ifdef` preprocessor macros (`FILL_TILE_WITH_FIRST_ELEMENT`, `FILL_WITH_VALUE`, `FILL_WITH_VALUE_FLOAT`) that are set based on input dtype at program factory lines 229-236. This allows the same kernel source to handle both bfloat16 and float32 data.

## Relevance to RMSNorm Design

### What Transfers Directly from BatchNorm

1. **CB creation pattern**: Using `create_cb()` from `cb_utils.hpp` with double-buffering.
2. **Compute kernel variants**: FPU vs SFPU selection based on `fp32_dest_acc_en`.
3. **`rsqrt_tile`** operation: Same SFPU call needed for `1/sqrt(mean(x^2) + eps)`.
4. **`binary_dest_reuse_tiles` pattern**: Useful for chaining multiply after rsqrt without intermediate CB.
5. **Epsilon tile pattern**: Fill once, persist for program lifetime.
6. **`_with_dt` helper functions** from `moreh_common.hpp`.
7. **Core distribution**: `split_work_to_cores()` for output tile assignment.

### What Must Be Different for RMSNorm

1. **No mean subtraction**: RMSNorm does not subtract mean, simplifying the formula.
2. **Reduction required**: RMSNorm must compute `mean(x^2)` which requires reducing along the last dimension. BatchNorm receives pre-computed statistics -- its compute kernel has NO reduction logic.
3. **Two-pass over input**: RMSNorm needs input tiles twice: once for computing `mean(x^2)`, once for multiplying by `1/sqrt(mean+eps)`. This requires either:
   - Input tiles persisting in CB across passes (needs larger CB or accumulation strategy), or
   - Re-reading input tiles from DRAM (simpler but slower).
4. **Reduction APIs**: Use `reduce_init/reduce_tile/reduce_uninit` from `api/compute/reduce.h`, or the fused `mul_reduce_scalar` API from `api/compute/experimental/mul_reduce_scalar.h`.
5. **Row-wise vs channel-wise**: BatchNorm broadcasts per-channel. RMSNorm reduces per-row (last dim). Work distribution should be per-row rather than per-output-tile.
6. **Tilize/untilize support**: For ROW_MAJOR input, use `tilize_init/tilize_block/tilize_uninit` and `pack_untilize_init/pack_untilize_block/pack_untilize_uninit` from layernorm's `layernorm_compute_utils.h`.

### Existing RMSNorm Reference Implementations

The deepseek_v3_b1 model contains a highly optimized rmsnorm implementation (`models/demos/deepseek_v3_b1/unified_kernels/rmsnorm.hpp`) that demonstrates:
1. **`mul_reduce_scalar` API**: Fused multiply-and-reduce that computes `sum(x * x)` across tiles in a single operation.
2. **`add_rsqrt_tile` API**: Fused `rsqrt(x + addend)` for computing `1/sqrt(mean + eps)`.
3. **`rmsnorm_mul_bcast_scalar_reuse_tiles`**: Specialized API for multiplying all tiles by a scalar from the dest register, reusing dest.
4. **`binary_dest_reuse_tiles` for gamma**: Multiply normalized result by weight.

The normalization `kernel_util/compute/numeric.h` provides reusable `row_wise_accumulate_with_epilogue` and `row_wise_mean` functions built on `reduce_tile`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What are the compute kernel APIs: tile_regs_acquire, tile_regs_commit, tile_regs_wait, tile_regs_release?"
   **Reason**: Understanding the dest register file synchronization protocol is essential for analyzing compute kernels.
   **Key Findings**: Four-phase protocol: acquire (math gets dst), commit (transfer to packer), wait (packer waits for dst ready), release (make dst available for next iteration). Unpack+math use acquire/commit, pack uses wait/release.

2. **Query**: "What is binary_dest_reuse_tiles and how does DEST_TO_SRCA work?"
   **Reason**: The FPU batch_norm kernel uses this pattern extensively for chaining operations.
   **Key Findings**: `binary_dest_reuse_tiles` avoids pack/unpack round-trips by keeping the previous operation's result in the dest register. `DEST_TO_SRCA` loads the dest value as SrcA and the CB tile as SrcB for the binary operation. The `_with_dt` suffix means the function handles data format reconfiguration for the unpack/math units.

3. **Query**: "What is rsqrt_tile in tt-metal compute kernels?"
   **Reason**: Core operation for both batch_norm and rmsnorm.
   **Key Findings**: SFPU operation computing element-wise `1/sqrt(x)` on a dest register tile. Signature: `rsqrt_tile<bool legacy_compat, bool FAST_APPROX>(uint32_t idst)`. Must call `rsqrt_tile_init()` first. Always an SFPU operation regardless of FPU/SFPU kernel variant.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Helper functions used by batch_norm compute kernel.
   **Key Information**: Defines `pack_tile_with_dt` (pack with data format reconfig when FP32_DEST_ACC_EN), `copy_tile_init_with_dt`, `add_tiles_init_with_dt`, `mul_tiles_init_with_dt`, `sub_tiles_init_with_dt`, and higher-level helpers like `mul_tiles_to_cb`, `add_tiles_to_cb`, etc.

2. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: RMSNorm needs reduction; batch_norm does not use it but understanding it is needed for the new operation.
   **Key Information**: `reduce_init<PoolType, ReduceDim>()`, `reduce_tile<PoolType, ReduceDim>(icb, icb_scaler, itile, itile_scaler, idst)`, `reduce_uninit()`. Requires a scaler CB. For SUM, scaler should be 1.0. Supports REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR.

3. **Source**: `tt_metal/hw/inc/api/compute/experimental/mul_reduce_scalar.h`
   **Reason**: Optimized fused multiply-reduce used in deepseek rmsnorm.
   **Key Information**: `mul_reduce_scalar_init(icb0, icb1)`, `mul_reduce_scalar_tile<PoolType>(icb0, icb1, num_tiles, scaler)`. Fuses `C = A * B` then `result = sum(all elements of C)`. Result stored at dest[0][0]. Processes up to 8 tiles.

4. **Source**: `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/add_rsqrt.h`
   **Reason**: Fused add+rsqrt used in deepseek rmsnorm.
   **Key Information**: `add_rsqrt_tile_init()`, `add_rsqrt_tile<fast_approx, vec_mode, ITERATIONS>(idst, addend)`. Computes `rsqrt(x + addend)` in a single SFPU call.

5. **Source**: `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/rmsnorm.h`
   **Reason**: Specialized rmsnorm broadcast-multiply API.
   **Key Information**: `rmsnorm_mul_bcast_scalar_reuse_tiles_init<num_tiles>(icb0)`, `rmsnorm_mul_bcast_scalar_reuse_tiles<num_tiles, clear_dest>(in_cb_id, in_tile_index, src_tile_index, dst_tile_index)`. Multiplies input tiles by a scalar from dest, reusing dest registers.

6. **Source**: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_compute_utils.h`
   **Reason**: Tilize/untilize patterns needed for ROW_MAJOR input support.
   **Key Information**: `tilize_row_major_block()`, `tilize_all_blocks_to_cb()`, `untilize_row_major_block()`, `untilize_all_blocks_from_cb()`. Use `tilize_init/tilize_block/tilize_uninit` and `pack_untilize_init/pack_untilize_block/pack_untilize_uninit`.

7. **Source**: `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h`
   **Reason**: Reusable row-wise reduction utilities for normalization.
   **Key Information**: `row_wise_accumulate_with_epilogue<reduce_type, reduce_dim, FLOAT32_REDUCTION>()` and `row_wise_mean()`. These handle blocked reduction with partial tile handling, scaler CBs, and input pop policies. Built on `reduce_init/reduce_tile/reduce_uninit`.

8. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Understanding how scalar constants are loaded into tiles.
   **Key Information**: `fill_with_val_bfloat16(cb_id, packed_scalar)` fills all 512 uint32 positions. `fill_with_val<ElementsV, ScalarT>(cb_id, scalar)` is a templated version. `fill_tile_with_first_element_bfloat16(cb_id)` reads first element and broadcasts to all positions.
