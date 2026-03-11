# Batch Norm Implementation Analysis

## Overview

Batch normalization computes the per-channel normalized output:

```
output = (input - batch_mean) / sqrt(batch_var + eps) [* weight] [+ bias]
```

The operation takes pre-computed batch_mean and batch_var tensors (1D per-channel) and applies normalization to a 4D input tensor. Optional affine transforms (weight/bias) are applied if provided.

**Program factory path**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Focus of this analysis**: Compute kernel structure, CB layout, multi-pass data reuse patterns, scalar/constant CB setup, and binary op broadcast patterns -- intended as a compute_core reference for a new layer_norm_rm operation.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Outer: channel groups (freq = Ht * Wt), Inner: tiles within channel group |

The compute kernel processes one output tile per inner-loop iteration. The key insight is a **frequency-based iteration pattern**: since batch_mean, batch_var, weight, and bias are per-channel scalars, they are reused across all spatial tiles (Ht * Wt) within a channel. The "freq" parameter (= cHt * cWt) determines how many input tiles share the same channel parameter tile.

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | batch_mean | batch_var | weight (optional) | bias (optional) |
|----------|-----------|------------|-----------|-------------------|-----------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] (or [N, C, 1, 1]) | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | same as input | same as input | same as input | same as input |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | Same as input [N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Configurable (defaults to input dtype) |

### Layout Transformations

The per-channel parameter tensors (batch_mean, batch_var, weight, bias) contain only one meaningful value per tile (since shape is [1,C,1,1]). The writer kernel reads these tiles and then calls `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast the single scalar value across all 1024 (or 32x32) elements of the tile. This creates a uniform tile that can be used directly in element-wise binary operations against full input tiles without requiring hardware broadcast instructions.

## Data Flow Pattern

### High-Level Flow

1. **Reader** reads input tiles from DRAM into `cb_input` (c_0). It also fills `cb_eps` (c_4) with the epsilon constant once at the start.
2. **Writer** (acting also as a secondary reader) reads batch_mean, batch_var, weight, and bias from DRAM, fills each tile with the first element (`FILL_TILE_WITH_FIRST_ELEMENT`), and pushes to their respective CBs. It also writes computed output tiles from `cb_output` (c_2) to DRAM.
3. **Compute** consumes all input CBs, performs the normalization math, and produces output tiles.

### Detailed Data Flow

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 0 (once) | Reader | Runtime arg (eps) | cb_eps (c_4) | `cb_reserve_back`, fill_with_val, `cb_push_back` |
| 1 | Reader | DRAM (input) | cb_input (c_0) | `cb_reserve_back`, `noc_async_read_tile`, `cb_push_back` |
| 2 | Writer | DRAM (batch_mean) | cb_batch_mean (c_1) | `cb_reserve_back`, `noc_async_read_tile`, `FILL_TILE_WITH_FIRST_ELEMENT`, `cb_push_back` |
| 3 | Writer | DRAM (batch_var) | cb_batch_var (c_3) | Same pattern as stage 2 |
| 4 | Writer | DRAM (weight) | cb_weight (c_5) | Same pattern, if weight_has_value |
| 5 | Writer | DRAM (bias) | cb_bias (c_6) | Same pattern, if bias_has_value |
| 6 | Compute | cb_batch_var, cb_eps | cb_den (c_7) | add_tiles + rsqrt_tile -> 1/sqrt(var+eps) |
| 7 | Compute | cb_input, cb_batch_mean | cb_tmp_1 or cb_output (c_8/c_2) | sub_tiles + binary_dest_reuse mul -> (x-mean)*inv_std |
| 8 | Compute | cb_tmp_1, cb_weight | cb_tmp_1 or cb_output | mul_tiles -> result*weight (if weight) |
| 9 | Compute | cb_tmp_1, cb_bias | cb_output (c_2) | add_tiles -> result+bias (if bias) |
| 10 | Writer | cb_output (c_2) | DRAM (output) | `cb_wait_front`, `noc_async_write_tile`, `cb_pop_front` |

### Reader/Writer Naming Caveat

The **writer** kernel serves a dual role: it reads batch_mean, batch_var, weight, and bias from DRAM (secondary reader function) AND writes the final output to DRAM. This is a **split reader pattern** -- the input tensor is read by the reader kernel, while auxiliary per-channel parameter tensors are read by the writer kernel to balance NoC traffic across both NoC0 and NoC1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tile staging | 2 tiles | 1 tile | Double | Reader | Compute | Block |
| c_1 | cb_batch_mean | Per-channel mean (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel-group (freq tiles) |
| c_2 | cb_output | Output tile staging | 2 tiles | 1 tile | Double | Compute | Writer | Block |
| c_3 | cb_batch_var | Per-channel variance (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel-group (1 per freq) |
| c_4 | cb_eps | Epsilon constant (all elements = eps) | 2 tiles | 1 tile | Double | Reader | Compute | Program (entire kernel) |
| c_5 | cb_weight | Per-channel weight (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel-group (freq tiles) |
| c_6 | cb_bias | Per-channel bias (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel-group (freq tiles) |
| c_7 | cb_den | Intermediate: 1/sqrt(var+eps) | 2 tiles | 1 tile | Double | Compute | Compute | Channel-group (freq tiles) |
| c_8 | cb_tmp_1 | Intermediate: normalized result before affine | 2 tiles | 1 tile | Double | Compute | Compute | Block |

### CB Aliasing for Conditional Affine Transforms

The compute kernel uses dynamic CB target selection based on whether weight/bias are present:

- `cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output` -- If no affine transform, normalized result goes directly to output CB.
- `cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output` -- If no bias but weight present, weight-scaled result goes directly to output CB.

This avoids unnecessary intermediate CB writes when affine transforms are not needed.

## Pipeline Pattern Summary

All CBs have capacity = 2 tiles and block size = 1 tile, enabling **double buffering** throughout. This allows the reader/writer to prepare the next tile while compute processes the current one.

## Multi-Pass Data Reuse Patterns (Critical for Layer Norm Reference)

### Channel-Group Broadcast Pattern

The most important pattern in this kernel is the **channel-group reuse**: per-channel parameters (batch_mean, batch_var, weight, bias, and computed cb_den) are loaded once and reused across `freq = Ht * Wt` spatial tiles within that channel.

**How it works in the compute kernel**:

1. At the start of each channel group, `batchnorm_bcast_tiles()` is called.
2. It first computes `1/sqrt(batch_var + eps)` and stores it in `cb_den`.
3. It then calls `cb_wait_front` on batch_mean, cb_den, weight, and bias -- these are held in their CBs.
4. An inner loop iterates over `freq` tiles, consuming one `cb_input` tile per iteration but reusing the same batch_mean/cb_den/weight/bias tile positions.
5. After all `freq` tiles are processed, `cb_pop_front` is called on batch_mean, cb_den, weight, and bias, signaling the writer to provide the next channel's parameters.

**CB lifetime implications**:
- `cb_eps` (c_4): **Program lifetime** -- pushed once by reader, waited on once at the start, popped once at the very end.
- `cb_batch_mean` (c_1), `cb_den` (c_7), `cb_weight` (c_5), `cb_bias` (c_6): **Channel-group lifetime** -- loaded once per channel, held during the inner spatial loop, popped after all spatial tiles for that channel are processed.
- `cb_input` (c_0), `cb_output` (c_2), `cb_tmp_1` (c_8): **Block (per-tile) lifetime** -- consumed/produced each iteration.

### Relevance to Layer Norm

For layer_norm_rm, the analogous pattern would be:
- **Per-row reuse**: mean and variance are computed per row (not per channel), so they would persist across the width dimension of a single row.
- **Epsilon CB**: Same program-lifetime pattern applies.
- **Gamma/beta (weight/bias)**: In layer norm, gamma and beta are per-feature (column) rather than per-channel, so the reuse pattern is reversed -- gamma/beta repeat across rows, not across spatial positions.

## Scalar/Constant CB Setup

### Epsilon (cb_eps, c_4)

The epsilon scalar is handled entirely by the **reader** kernel:

1. The packed epsilon value is passed as runtime arg 0 (`packed_scalar_eps`).
2. For FLOAT32: `std::bit_cast<uint32_t>(scalar)` -- bit-cast float to uint32.
3. For BFLOAT16: `pack_two_bfloat16_into_uint32({scalar, scalar})` -- pack two bfloat16 values.
4. In the reader kernel, `cb_reserve_back(cb_id_eps, 1)` is called.
5. `FILL_WITH_VALUE_FLOAT` (fp32) or `FILL_WITH_VALUE` (bf16) fills all 1024 elements of the tile with the eps value.
6. `cb_push_back(cb_id_eps, 1)` makes it available to compute.
7. Compute calls `cb_wait_front(cb_eps, 1)` once at the start and `cb_pop_front(cb_eps, 1)` once at the very end.

The fill functions are from `fill_tile_utils.hpp`:
- `fill_with_val<1024, float>(cb_id, scalar.f)` -- writes 1024 floats (4096 bytes, a full fp32 tile)
- `fill_with_val_bfloat16(cb_id, packed_scalar)` -- writes 512 uint32s (1024 bytes, a full bf16 tile)

### Per-Channel Parameters (batch_mean, batch_var, weight, bias)

These are handled by the **writer** kernel using `FILL_TILE_WITH_FIRST_ELEMENT`:
1. A tile is read from DRAM via `noc_async_read_tile`.
2. `FILL_TILE_WITH_FIRST_ELEMENT(cb_id)` reads the first element of the tile and fills all positions with that value.
3. This converts a [1,C,1,1] parameter tile (where only position [0,0] is meaningful) into a uniform tile suitable for element-wise operations.

For bf16: `fill_tile_with_first_element_bfloat16` reads uint16 at position 0, packs to uint32, fills 512 uint32s.
For fp32: `fill_tile_with_first_element<float>` reads float at position 0, fills 1024 floats.

## Compute Kernel Structure (Primary Analysis Focus)

### Two Variants

The program factory selects between two compute kernel files based on `fp32_dest_acc_en`:
- **FPU path**: `batch_norm_kernel.cpp` -- used when `fp32_dest_acc_en = false`
- **SFPU path**: `batch_norm_sfpu_kernel.cpp` -- used when `fp32_dest_acc_en = true`

### kernel_main() Structure (Both Variants)

```
kernel_main():
  1. Read runtime args: num_tiles, tile_freq, tile_start
  2. Read compile-time args: weight_has_value, bias_has_value, all CB IDs
  3. Initialize: binary_op_init_common (FPU) or unary_op_init_common (SFPU)
  4. Compute iteration counts:
     - complete_iterations = (num_tiles + tile_start) / tile_freq
     - remaining_iterations = (num_tiles + tile_start) % tile_freq
  5. Wait for epsilon tile: cb_wait_front(cb_eps, 1)  [held for entire program]
  6. Loop over complete channel groups:
     for i in 0..complete_iterations:
       batchnorm_bcast_tiles(freq=tile_freq, tile_start=tile_start on first, 0 after)
  7. Handle remaining tiles (partial channel group):
     if remaining_iterations > 0:
       batchnorm_bcast_tiles(freq=remaining_iterations, tile_start)
  8. Pop epsilon: cb_pop_front(cb_eps, 1)
```

### batchnorm_bcast_tiles() -- FPU Variant (Detailed)

**Signature**:
```cpp
void batchnorm_bcast_tiles(
    uint32_t cb_bcast,      // cb_batch_mean (c_1) -- channel parameter
    uint32_t cb_other,      // cb_input (c_0) -- input tiles
    uint32_t freq,          // number of spatial tiles in this channel group
    uint32_t tile_start,    // starting tile offset within channel group
    uint32_t cb_batch_var,  // c_3
    uint32_t cb_eps,        // c_4
    uint32_t cb_den,        // c_7 -- intermediate for 1/sqrt(var+eps)
    uint32_t cb_weight,     // c_5
    uint32_t cb_bias,       // c_6
    uint32_t cb_tmp_1,      // c_8
    uint32_t cb_output_0,   // c_2
    uint32_t weight_has,
    uint32_t bias_has)
```

**Phase 1: Compute inverse standard deviation** (once per channel group)

```cpp
// 1/(sqrt(batch_var + eps))
cb_reserve_back(cb_den, 1);
cb_wait_front(cb_batch_var, 1);

tile_regs_acquire();
add_tiles_init_with_dt(cb_batch_var, cb_eps);    // Signature: void add_tiles_init_with_dt(uint32_t icb0, uint32_t icb1)
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);     // Signature: void add_tiles(uint32_t cb0, uint32_t cb1, uint32_t tile0, uint32_t tile1, uint32_t dst)
rsqrt_tile_init();                                 // Signature: void rsqrt_tile_init()
rsqrt_tile(dst0);                                  // Signature: void rsqrt_tile(uint32_t idst) -- operates on DST in-place
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(dst0, cb_den);                  // Signature: void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb)
tile_regs_release();

cb_pop_front(cb_batch_var, 1);
cb_push_back(cb_den, 1);
```

**Key observation**: `add_tiles` takes two CB tile positions as SrcA and SrcB, computes var+eps in DST. Then `rsqrt_tile` applies 1/sqrt in-place on DST. This is an example of **operation chaining within a single acquire/commit block** -- avoiding an intermediate pack/unpack cycle.

**Phase 2: Wait for all channel-group parameters**

```cpp
cb_wait_front(cb_bcast, 1);       // batch_mean -- held for entire inner loop
cb_wait_front(cb_den, 1);         // inv_std -- held for entire inner loop
if (weight_has_value) cb_wait_front(cb_weight, 1);
if (bias_has_value) cb_wait_front(cb_bias, 1);
```

**Phase 3: Inner loop over spatial tiles** (freq iterations)

For each spatial tile j in [tile_start, freq):

**Step 3a: Subtract mean and multiply by inv_std (fused in FPU path)**

```cpp
cb_wait_front(cb_other, 1);            // wait for input tile
cb_reserve_back(cb_affine_or_out, 1);  // reserve output or temp

tile_regs_acquire();
sub_tiles_init(cb_other, cb_bcast);
sub_tiles(cb_other, cb_bcast, 0, 0, 0);   // DST[0] = input - batch_mean

// KEY API: binary_dest_reuse_tiles
binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
// This loads DST[0] -> SrcA, unpacks cb_den -> SrcB, then DST[0] = SrcA * SrcB
// Result: DST[0] = (input - mean) * inv_std
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(0, cb_affine_or_out);
tile_regs_release();

cb_push_back(cb_affine_or_out, 1);
cb_pop_front(cb_other, 1);   // pop input tile (consumed)
// NOTE: cb_bcast (batch_mean) and cb_den are NOT popped -- reused for next spatial tile
```

**Step 3b: Multiply by weight (optional)**

```cpp
if (weight_has_value) {
    cb_reserve_back(cb_scaled_output, 1);
    cb_wait_front(cb_affine_or_out, 1);

    tile_regs_acquire();
    mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
    mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_scaled_output);
    tile_regs_release();

    cb_pop_front(cb_affine_or_out, 1);
    cb_push_back(cb_scaled_output, 1);
}
```

**Step 3c: Add bias (optional)**

```cpp
if (bias_has_value) {
    cb_reserve_back(cb_output_0, 1);
    cb_wait_front(cb_tmp_1, 1);

    tile_regs_acquire();
    add_tiles_init_with_dt(cb_tmp_1, cb_bias);
    add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_output_0);
    tile_regs_release();

    cb_pop_front(cb_tmp_1, 1);
    cb_push_back(cb_output_0, 1);
}
```

**Phase 4: Release channel-group parameters**

```cpp
cb_pop_front(cb_bcast, 1);   // release batch_mean
cb_pop_front(cb_den, 1);     // release inv_std
if (weight_has_value) cb_pop_front(cb_weight, 1);
if (bias_has_value) cb_pop_front(cb_bias, 1);
```

### binary_dest_reuse_tiles -- Key API Explained

`binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_den, tile_idx, dst_idx)`:

- **Purpose**: Performs an element-wise binary operation where one operand comes from the DST register (already computed) and the other from a CB.
- **DEST_TO_SRCA**: The current DST[dst_idx] value is moved to SrcA register. The tile from cb_den at tile_idx is unpacked into SrcB. The multiplication result is written back to DST[dst_idx].
- **Why used here**: After `sub_tiles` puts (input - mean) into DST[0], we want to multiply that by inv_std from cb_den without packing to a CB and unpacking again. `binary_dest_reuse_tiles` avoids the round-trip through L1, saving both latency and a CB slot.
- **Init signature**: `binary_dest_reuse_tiles_init<EltwiseBinaryType, EltwiseBinaryReuseDestType>(uint32_t icb0)`
- **Op signature**: `binary_dest_reuse_tiles<EltwiseBinaryType, EltwiseBinaryReuseDestType>(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)`

### SFPU Variant Differences

The SFPU variant (`batch_norm_sfpu_kernel.cpp`) uses explicit `copy_tile` operations instead of FPU binary ops:

1. **Initialization**: `unary_op_init_common(cb_other, cb_output_0)` instead of `binary_op_init_common`.
2. **Add (var + eps)**: Uses `copy_tile` to load both operands into adjacent DST registers (dst0 and dst0+1), then `add_binary_tile(dst0, dst0+1, dst0)`.
3. **Sub (input - mean)**: Uses `copy_tile(cb_other, i, i*2)` and `copy_tile(cb_bcast, i, i*2+1)`, then `sub_binary_tile(i*2, i*2+1, i*2)`.
4. **Mul**: Uses `copy_tile` + `mul_binary_tile` instead of `mul_tiles` or `binary_dest_reuse_tiles`.
5. **DST register usage**: Uses indices `i*2` and `i*2+1` (0 and 1) to hold both operands in DST simultaneously.
6. **copy_tile_to_dst_init_short_with_dt**: Called before each `copy_tile` sequence to reconfigure unpacker for the correct data format.
7. **rsqrt_tile**: Same API in both variants -- it's an SFPU operation regardless.

**Key difference**: The SFPU path explicitly manages DST register allocation (using even/odd pairs), while the FPU path uses the hardware unpack mechanism to load CB data directly into SrcA/SrcB registers.

## Index Calculations

### Frequency-Based Channel Grouping

The compute kernel receives three runtime args:
- `num_tiles`: total tiles this core must process
- `tile_freq`: = `cHt * cWt` -- number of spatial tiles per channel
- `tile_start`: = `start_tile_id % cHtWt` -- offset within first channel group (for partial starts)

The iteration calculation:
```cpp
complete_iterations = (num_tiles + tile_start) / tile_freq   // full channel groups
remaining_iterations = (num_tiles + tile_start) % tile_freq  // partial trailing group
```

The first invocation of `batchnorm_bcast_tiles` uses `tile_start` as the starting offset within the channel group. After the first complete iteration, `tile_start` resets to 0 for subsequent iterations (`tile_start = 0` in the loop update).

### Reader Index Mapping

The reader uses N/C/HtWt decomposition:
```cpp
tile_offset = start_n * n_stride + start_c * c_stride + start_t
```
Where:
- `n_stride = aHt * aWt * aC` (if N > 1, else 0)
- `c_stride = aHt * aWt` (if C > 1, else 0)
- Tiles are traversed in N -> C -> spatial order

## Memory Access Patterns

### Read Pattern
- **Input tiles**: Sequential within each channel, with stride jumps between channels and batches. One tile per reader loop iteration.
- **Per-channel parameters**: One tile per channel, with `FILL_TILE_WITH_FIRST_ELEMENT` broadcast. Read once per freq spatial tiles.
- **Epsilon**: Single tile read once at program start.

### Write Pattern
- **Output tiles**: Sequential write of one tile per inner-loop iteration, using linear tile IDs: `start_tile_id + num_tiles_written`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (row-major traversal) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | `num_cores_x * num_cores_y` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split via `split_work_to_cores` |

Work distribution uses `tt::tt_metal::split_work_to_cores()` which divides `num_output_tiles` across available cores. Cores in `core_group_1` get `num_tiles_per_core_group_1` tiles; cores in `core_group_2` get `num_tiles_per_core_group_2` tiles (one fewer). Cores outside both groups receive zero-initialized args and early-return.

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t | 1 if weight tensor is provided, 0 otherwise |
| 1 | bias_has_value | uint32_t | 1 if bias tensor is provided, 0 otherwise |
| 2 | cb_input | uint32_t | CB ID for input tensor (c_0) |
| 3 | cb_batch_mean | uint32_t | CB ID for batch mean (c_1) |
| 4 | cb_output | uint32_t | CB ID for output tensor (c_2) |
| 5 | cb_batch_var | uint32_t | CB ID for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB ID for epsilon constant (c_4) |
| 7 | cb_den | uint32_t | CB ID for 1/sqrt(var+eps) intermediate (c_7) |
| 8 | cb_weight | uint32_t | CB ID for weight tensor (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB ID for intermediate result (c_8) |
| 10 | cb_bias | uint32_t | CB ID for bias tensor (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total number of output tiles this core processes |
| 1 | tile_freq | uint32_t | Tiles per channel group (= Ht * Wt), controls parameter reuse |
| 2 | tile_start | uint32_t | Starting offset within first channel group (= start_tile_id % freq) |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value packed as uint32 (bit-cast float or packed bfloat16 pair) |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | Starting output tile index for this core |
| 3 | num_tiles | uint32_t | Number of tiles to process |
| 4 | HtWt | uint32_t | Spatial tiles per channel (= cHt * cWt) |
| 5 | n_stride | uint32_t | Tile stride between batches (0 if N=1) |
| 6 | c_stride | uint32_t | Tile stride between channels (0 if C=1) |
| 7 | N | uint32_t | Number of batches |
| 8 | C | uint32_t | Number of channels |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Batch mean tensor buffer address |
| 1 | batch_var_addr | uint32_t | Batch variance tensor buffer address |
| 2 | weight_addr | uint32_t | Weight tensor buffer address (0 if not present) |
| 3 | bias_addr | uint32_t | Bias tensor buffer address (0 if not present) |
| 4 | dst_addr | uint32_t | Output tensor buffer address |
| 5 | start_tile_id | uint32_t | Starting output tile index |
| 6 | num_tiles | uint32_t | Number of tiles to process |
| 7 | HtWt | uint32_t | Spatial tiles per channel |
| 8 | n_stride | uint32_t | Batch stride for parameter tensors |
| 9 | c_stride | uint32_t | Channel stride for parameter tensors |
| 10 | N | uint32_t | Batches |
| 11 | C | uint32_t | Channels |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM (input) | c_0 (input), c_4 (eps) | Read input tiles, fill epsilon tile |
| writer_batch_norm | RISCV_1 | NOC1 | DRAM (mean/var/weight/bias), c_2 (output) | c_1, c_3, c_5, c_6, DRAM (output) | Read params with broadcast-fill, write output tiles |
| batch_norm_kernel | Compute (FPU) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2, c_7, c_8 | add, rsqrt, sub, binary_dest_reuse mul, mul, add |
| batch_norm_sfpu_kernel | Compute (SFPU) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2, c_7, c_8 | copy_tile, add_binary, sub_binary, mul_binary, rsqrt |

### Compute Kernel Key Logic

**FPU variant** (`batch_norm_kernel.cpp`):
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`
- Uses `binary_op_init_common(cb_other, cb_bcast, cb_output_0)` for initialization.
- Chains `sub_tiles` + `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` to fuse subtraction and multiplication within one acquire/commit block.
- Standard `add_tiles` / `mul_tiles` with `_init_with_dt` variants for FP32 dest accumulation support.

**SFPU variant** (`batch_norm_sfpu_kernel.cpp`):
- **File**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`
- Uses `unary_op_init_common(cb_other, cb_output_0)` for initialization.
- All binary operations use explicit `copy_tile` to DST followed by `{add,sub,mul}_binary_tile`.
- `copy_tile_to_dst_init_short_with_dt(cb_a, cb_b)` reconfigures unpacker between different CB data formats.
- Uses DST register pairs (i*2, i*2+1) for two-operand operations.

## Implementation Notes

### FP32 Destination Accumulation Mode

When `fp32_dest_acc_en = true`:
1. The SFPU kernel variant is selected.
2. `UnpackToDestMode::UnpackToDestFp32` is set for all CBs that participate in compute.
3. All `_with_dt` helper functions (from `moreh_common.hpp`) call `reconfig_data_format()` or `pack_reconfig_data_format()` to handle format conversion between L1 (bf16) and DST (fp32).

### pack_tile_with_dt Explained

From `moreh_common.hpp`:
```cpp
ALWI void pack_tile_with_dt(uint32_t ifrom_dst, uint32_t icb) {
#if defined FP32_DEST_ACC_EN
    pack_reconfig_data_format(icb);   // reconfigure packer for target CB's format
#endif
    pack_tile(ifrom_dst, icb);        // standard pack from DST to CB
}
```
The `_with_dt` (data type) suffix means the function handles data format reconfiguration when FP32 dest accumulation is enabled. Without it, the packer might try to pack fp32 DST data into a bf16 CB without proper conversion.

### Conditional Defines for Data Type

The program factory sets preprocessor defines based on input dtype:
- **FLOAT32**: `FILL_TILE_WITH_FIRST_ELEMENT = fill_tile_with_first_element<float>`, `FILL_WITH_VALUE_FLOAT = fill_with_val<1024, float>`
- **BFLOAT16**: `FILL_TILE_WITH_FIRST_ELEMENT = fill_tile_with_first_element_bfloat16`, `FILL_WITH_VALUE = fill_with_val_bfloat16`

### Tile Register Protocol Summary

Every compute operation follows the strict protocol:
1. `tile_regs_acquire()` -- acquire DST registers (zeroes them)
2. Unpack + Math operations (e.g., `add_tiles`, `sub_tiles`, `rsqrt_tile`)
3. `tile_regs_commit()` -- hand ownership to packer
4. `tile_regs_wait()` -- packer waits for data
5. `pack_tile_with_dt(dst, cb)` -- pack from DST to CB
6. `tile_regs_release()` -- release DST registers

Multiple math operations can be chained between acquire and commit (e.g., sub_tiles then binary_dest_reuse_tiles) as long as they operate on DST registers.

### Helper Functions Available from moreh_common.hpp (Relevant for Layer Norm)

Key helpers and their signatures from `moreh_common.hpp`:

| Helper | Signature | Description |
|--------|-----------|-------------|
| `pack_tile_with_dt` | `(uint32_t ifrom_dst, uint32_t icb)` | Pack with data format reconfig |
| `copy_tile_init_with_dt` | `(uint32_t icb, uint32_t transpose = 0)` | Copy init with format reconfig |
| `add_tiles_init_with_dt` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` | Add init with format reconfig |
| `sub_tiles_init_with_dt` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` | Sub init with format reconfig |
| `mul_tiles_init_with_dt` | `(uint32_t icb0 = 0, uint32_t icb1 = 1)` | Mul init with format reconfig |
| `mul_tiles_to_cb` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` | Full multiply pipeline with CB ops |
| `add_tiles_to_cb` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` | Full add pipeline with CB ops |
| `sub_tiles_to_cb` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` | Full sub pipeline with CB ops |
| `copy_tile_to_cb` | `(icb, ocb, itile=0, pop=1)` | Copy tile between CBs |
| `recip_tile_to_cb` | `(icb, ocb, itile=0, pop=1)` | 1/x with full CB pipeline |
| `mul_tiles_bcast_rows_to_cb` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` | Multiply with row broadcast |
| `sub_tiles_bcast_cols_to_cb` | `(icb0, icb1, ocb, itile0=0, itile1=0, pop0=1, pop1=1)` | Subtract with column broadcast |

### Reduce API (Not Used in Batch Norm, But Critical for Layer Norm)

From `api/compute/reduce.h`, the reduce tile API:

```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb);

template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst);

template <bool enforce_fp32_accumulation = false>
void reduce_uninit(uint32_t icb = 0);
```

- `PoolType`: `SUM`, `AVG`, `MAX`
- `ReduceDim`: `REDUCE_ROW` (reduce across columns, result in column), `REDUCE_COL` (reduce across rows, result in row), `REDUCE_SCALAR` (reduce both dims)
- `icb_scaler`: A CB containing the scaling factor tile. For SUM use 1.0, for AVG use 1/N. The scaler tile must have values in the first row of each face.
- **Important**: `reduce_uninit()` must be called after reduce operations before other operations, to reset the packer edge mask.

The `generate_reduce_scaler` helper (from `ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp`) fills a tile suitable for reduce_tile's scaler CB:
```cpp
template <bool half_tile = false>
void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler);
```
It zeros the tile, then writes the packed scaler value to the first row (8 elements) of each face.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How do tile_regs_acquire/commit/wait/release work and what is the FPU vs SFPU difference?"
   **Reason**: Needed to understand the DST register synchronization protocol used throughout the compute kernel.
   **Key Findings**: tile_regs_acquire() acquires DST registers and zeroes them. The protocol is acquire -> math -> commit -> wait -> pack -> release. FPU operations (add_tiles, sub_tiles, mul_tiles) unpack from CBs directly to SrcA/SrcB. SFPU operations require explicit copy_tile to load data into DST first. The `_with_dt` variants handle data format reconfiguration for FP32 accumulation mode.

2. **Query**: "What is binary_dest_reuse_tiles and how does DEST_TO_SRCA work?"
   **Reason**: This API is central to the FPU kernel's fused subtract-multiply pattern.
   **Key Findings**: (Query failed, but found documentation in `eltwise_binary.h` header comments.) DEST_TO_SRCA moves the current DST[idst] value to SrcA, then unpacks the specified CB tile into SrcB (or vice versa for DEST_TO_SRCB). The binary operation result goes back to DST[idst]. This avoids a pack-unpack round-trip when chaining operations.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary.h` (lines 222-257)
   **Reason**: Needed exact API documentation for binary_dest_reuse_tiles.
   **Key Information**: Full documentation of DEST_TO_SRCA and DEST_TO_SRCB modes, confirming that the DST value is loaded into the source register specified, and the CB tile goes to the other source register.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h`
   **Reason**: Needed to understand rsqrt_tile API.
   **Key Information**: `rsqrt_tile(idst)` computes element-wise 1/sqrt(x) on DST[idst] in-place. Uses SFPU math regardless of kernel variant.

3. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: The batch_norm compute kernel includes this header; needed to understand all available helper functions.
   **Key Information**: Provides `_with_dt` wrappers for all common operations that handle FP32 dest accumulation format reconfiguration. Also provides composite `_to_cb` helpers that encapsulate the full acquire/math/commit/wait/pack/release + CB ops pipeline.

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Needed to understand how scalar values are broadcast into full tiles.
   **Key Information**: `fill_with_val_bfloat16` writes 512 uint32s (packed pairs). `fill_with_val<1024, float>` writes 1024 floats. `fill_tile_with_first_element_bfloat16` reads position 0 and broadcasts to all 1024 elements.

5. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: Layer norm will need reduce operations; documenting the API for downstream reference.
   **Key Information**: `reduce_init/reduce_tile/reduce_uninit` triplet. Requires a scaler CB. ReduceDim options: REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR. Must call reduce_uninit before next non-reduce operation.

6. **Source**: `ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp`
   **Reason**: Understanding how to set up the scaler CB required by reduce_tile.
   **Key Information**: Zeros the tile first, then writes the packed scaler to the first 8 elements (first row) of each face. For AVG reduction, scaler should be 1/N.

7. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the create_cb helper used in the program factory.
   **Key Information**: `create_cb(cb_id, program, core_spec, page_size, num_pages, data_format)` creates a CircularBuffer with the given configuration and returns `tuple<uint32_t, CBHandle>`.
