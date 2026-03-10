# Batch Norm Implementation Analysis (Compute Core Reference)

## Overview

Batch normalization computes: `output = gamma * (input - mean) / sqrt(var + eps) + beta` on a per-channel basis. Unlike layer norm, the mean and variance are **pre-computed** (supplied as input tensors), so the compute kernel only performs normalization and optional affine transform -- it does not compute reductions.

**Program factory**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`

**Compute kernel (FPU)**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`

**Compute kernel (SFPU, fp32)**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_sfpu_kernel.cpp`

**Reader**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp`

**Writer**: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp`

---

## Relevance as Compute Core Reference for layer_norm_rm

This analysis focuses on patterns directly transferable to a layer_norm_rm operation:
1. **CB layout for intermediates** -- how scratchpad CBs (cb_den, cb_tmp_1) are allocated and reused
2. **Multi-pass data reuse** -- which CBs persist across the inner loop vs. per-iteration
3. **Scalar/constant CB setup** -- how epsilon is broadcast-filled into a full tile
4. **Binary op with broadcast** -- how a 1-value-per-channel parameter is applied to every tile in that channel
5. **Affine transform (gamma/beta)** -- how optional weight/bias scaling is conditionally chained
6. **FPU vs SFPU kernel variant** -- how fp32_dest_acc_en selects the kernel and what changes

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Two-level: outer loop over "channel groups" (complete_iterations), inner loop over tiles within a channel group (freq tiles per group) |

The work is split across output tiles linearly. Each core gets a contiguous block of `num_tiles_per_core` output tiles to process. The tiles are logically ordered N-C-Ht-Wt (batch-channel-height_tiles-width_tiles).

---

## Tensor Format and Layout

### Input Tensors

| Property | Input (x) | batch_mean | batch_var | weight (gamma) | bias (beta) |
|----------|-----------|------------|-----------|----------------|-------------|
| **Logical shape** | [N, C, H, W] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] | [1, C, 1, 1] |
| **Dimension convention** | NCHW | NCHW | NCHW | NCHW | NCHW |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM | DRAM | DRAM | DRAM | DRAM |
| **Data type** | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 | BFLOAT16 or FLOAT32 |

### Output Tensor

| Property | Output |
|----------|--------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM |
| **Data type** | Same as input (or specified dtype) |

### Critical: Per-Channel Parameter Broadcasting

The per-channel parameters (mean, var, weight, bias) have shape [1, C, 1, 1]. The **writer** kernel reads one tile per channel from these tensors and calls `FILL_TILE_WITH_FIRST_ELEMENT(cb_id)` to broadcast the single scalar value across the entire 32x32 tile. This is the mechanism for broadcasting a per-channel scalar to all spatial locations within that channel.

---

## Data Flow Pattern

### High-Level Flow

```
Reader:  reads input tiles     --> cb_input (c_0)
Reader:  fills eps scalar tile --> cb_eps (c_4)

Writer:  reads mean tiles      --> cb_batch_mean (c_1)    [fills with first element]
Writer:  reads var tiles       --> cb_batch_var (c_3)     [fills with first element]
Writer:  reads weight tiles    --> cb_weight (c_5)        [fills with first element]
Writer:  reads bias tiles      --> cb_bias (c_6)          [fills with first element]

Compute: cb_batch_var + cb_eps          --> cb_den       (var + eps)
Compute: rsqrt(cb_den)                  --> cb_den       (1/sqrt(var + eps))
Compute: cb_input - cb_batch_mean       --> dst register
Compute: dst * cb_den                   --> cb_tmp_1 or cb_output
Compute: cb_tmp_1 * cb_weight           --> cb_tmp_1 or cb_output   (if gamma)
Compute: cb_tmp_1 + cb_bias             --> cb_output               (if beta)

Writer:  reads cb_output (c_2) --> DRAM
```

### Important Naming Caveat

The **writer** kernel handles reading mean, var, weight, and bias from DRAM (not the reader). This is a deliberate split: the reader handles the high-bandwidth input tensor, while the writer handles the low-bandwidth per-channel parameters that change only once per channel group. This is common in normalization ops and reflects the principle that the "reader/writer" names refer to core assignment (RISC-V 0 vs 1), not exclusively to read vs write function.

### Two-Level Loop Structure (Critical for layer_norm_rm Reference)

The compute kernel organizes work into **channel groups**. The variable `freq = cHt * cWt` represents the number of tiles in one channel's spatial extent (Ht * Wt). The variable `counter = start_tile_id % cHtWt` represents the starting offset within the first channel group for this core.

```
complete_iterations = (num_tiles + tile_start) / tile_freq
remaining_iterations = (num_tiles + tile_start) % tile_freq
```

**Outer loop** (per channel group): Per-channel values (mean, var, weight, bias) are loaded once and persist across all spatial tiles within that channel.

**Inner loop** (per tile within channel): For each spatial tile, the compute kernel subtracts mean, multiplies by inv_sqrt, and optionally applies gamma/beta.

This is the **multi-pass data reuse pattern**: per-channel parameters are loaded once for `freq` tiles of computation.

---

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tiles from DRAM | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_1 | cb_batch_mean | Per-channel mean (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group |
| c_2 | cb_output | Normalized output tiles | 2 tiles | 1 tile | Double | Compute | Writer | Block (per tile) |
| c_3 | cb_batch_var | Per-channel variance (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group |
| c_4 | cb_eps | Epsilon scalar (full tile) | 2 tiles | 1 tile | Double | Reader | Compute | Program (entire kernel) |
| c_5 | cb_weight | Per-channel gamma (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group |
| c_6 | cb_bias | Per-channel beta (broadcast-filled) | 2 tiles | 1 tile | Double | Writer | Compute | Channel group |
| c_7 | cb_den | Intermediate: 1/sqrt(var+eps) | 2 tiles | 1 tile | Double | Compute | Compute | Channel group |
| c_8 | cb_tmp_1 | Intermediate: (x-mean)*invstd or scaled result | 2 tiles | 1 tile | Double | Compute | Compute | Block (per tile) |

### CB Lifetime Analysis (Key for layer_norm_rm)

**Program lifetime** (persists across entire kernel execution):
- `cb_eps` (c_4): Filled once by reader at startup. Compute does `cb_wait_front(cb_eps, 1)` at the beginning and `cb_pop_front(cb_eps, 1)` only at the very end of `kernel_main()`. This tile lives for the entire program.

**Channel-group lifetime** (persists across all tiles in one channel):
- `cb_batch_mean` (c_1): Loaded once per channel by writer; popped once after all spatial tiles processed.
- `cb_den` (c_7): Computed once per channel by compute (from var+eps); popped once after all spatial tiles processed.
- `cb_weight` (c_5): Loaded once per channel by writer (if present); popped after all spatial tiles.
- `cb_bias` (c_6): Loaded once per channel by writer (if present); popped after all spatial tiles.
- `cb_batch_var` (c_3): Loaded once per channel by writer; consumed immediately by the rsqrt computation at the start of the channel group.

**Per-tile lifetime** (produced and consumed each iteration):
- `cb_input` (c_0): One tile in, one tile consumed per iteration.
- `cb_output` (c_2): One tile produced, one tile written out per iteration.
- `cb_tmp_1` (c_8): Intermediate produced and consumed within the same tile iteration.

### CB Routing for Conditional Affine Transform

The compute kernel uses clever CB aliasing based on whether gamma/beta are present:

```cpp
auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
```

This means:
- **No gamma, no beta**: normalized result goes directly to `cb_output_0`
- **Gamma only**: normalized result goes to `cb_tmp_1`, then gamma-multiply writes to `cb_output_0`
- **Gamma and beta**: normalized result goes to `cb_tmp_1`, gamma-multiply writes to `cb_tmp_1` (reuse), beta-add writes to `cb_output_0`
- **Beta only**: normalized result goes to `cb_tmp_1`, beta-add writes to `cb_output_0`

---

## Compute Kernel Structure (FPU Variant -- Primary Reference)

File: `batch_norm_kernel.cpp`

### Initialization

```cpp
binary_op_init_common(cb_other, cb_bcast, cb_output_0);
```

This is the general HW initialization for all binary compute operations. It configures the unpacker, math unit, and packer. Called once at kernel start, not per-tile.

### Main Function Structure

```cpp
void kernel_main() {
    // 1. Get runtime args
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);   // = Ht * Wt (tiles per channel)
    uint32_t tile_start = get_arg_val<uint32_t>(2);   // offset into first channel

    // 2. Get compile-time args (CB IDs, feature flags)
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;
    // CB IDs at compile-time args 2-10

    // 3. Initialize HW
    binary_op_init_common(cb_other, cb_bcast, cb_output_0);

    // 4. Wait for epsilon (program-lifetime CB)
    cb_wait_front(cb_eps, onetile);

    // 5. Loop: complete channel groups
    for (i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(...);
    }

    // 6. Remainder tiles (partial channel group)
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(...);
    }

    // 7. Release epsilon
    cb_pop_front(cb_eps, onetile);
}
```

### batchnorm_bcast_tiles Function (FPU Variant)

This is the core computation for one channel group. Signature:

```cpp
ALWI void batchnorm_bcast_tiles(
    uint32_t cb_bcast,      // cb_batch_mean (c_1)
    uint32_t cb_other,      // cb_input (c_0)
    uint32_t freq,          // number of tiles to process in this group
    uint32_t tile_start,    // starting tile offset within group
    uint32_t cb_batch_var,  // c_3
    uint32_t cb_eps,        // c_4
    uint32_t cb_den,        // c_7, intermediate
    uint32_t cb_weight,     // c_5, optional gamma
    uint32_t cb_bias,       // c_6, optional beta
    uint32_t cb_tmp_1,      // c_8, intermediate
    uint32_t cb_output_0,   // c_2, output
    uint32_t weight_has,    // bool
    uint32_t bias_has)      // bool
```

#### Phase 1: Compute inv_sqrt (once per channel group)

```cpp
// var + eps
cb_reserve_back(cb_den, onetile);
cb_wait_front(cb_batch_var, onetile);

tile_regs_acquire();
add_tiles_init_with_dt(cb_batch_var, cb_eps);      // init add HW for these CB data formats
add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);       // dst0 = var + eps
rsqrt_tile_init();                                  // init SFPU rsqrt
rsqrt_tile(dst0);                                   // dst0 = 1/sqrt(var + eps)
tile_regs_commit();

tile_regs_wait();
pack_tile_with_dt(dst0, cb_den);                   // pack dst0 -> cb_den
tile_regs_release();

cb_pop_front(cb_batch_var, onetile);               // done with var for this channel
cb_push_back(cb_den, onetile);                     // cb_den now holds inv_sqrt
```

**Key observations**:
- `add_tiles` is an FPU binary op: takes two CB indices + tile indices + dst register index.
- `rsqrt_tile` is an SFPU unary op that operates on the dst register in-place.
- FPU and SFPU ops can be chained within the same `tile_regs_acquire/commit` block.
- `add_tiles_init_with_dt` is a wrapper (from `moreh_common.hpp`) that calls `reconfig_data_format(icb0, icb1)` when `FP32_DEST_ACC_EN` is defined, then calls `add_tiles_init(icb0, icb1)`.

#### Phase 2: Wait for per-channel parameters (once per channel group)

```cpp
cb_wait_front(cb_bcast, onetile);    // mean -- stays until all spatial tiles done
cb_wait_front(cb_den, onetile);      // inv_sqrt -- stays until all spatial tiles done
if (weight_has_value) cb_wait_front(cb_weight, onetile);
if (bias_has_value) cb_wait_front(cb_bias, onetile);
```

These are NOT popped until the end of the channel group. They persist across the entire inner loop.

#### Phase 3: Per-tile normalization (inner loop)

```cpp
for (uint32_t j = tile_start; j < freq; ++j) {
    // --- Step A: (input - mean) * inv_sqrt ---
    cb_wait_front(cb_other, onetile);          // wait for input tile
    cb_reserve_back(cb_affine_or_out, onetile);

    tile_regs_acquire();
    sub_tiles_init(cb_other, cb_bcast);
    sub_tiles(cb_other, cb_bcast, 0, 0, 0);   // dst0 = input - mean

    // Dest reuse: multiply dst0 by inv_sqrt without re-reading result from CB
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
    // dst0 = (input - mean) * inv_sqrt
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(0, cb_affine_or_out);
    tile_regs_release();

    cb_push_back(cb_affine_or_out, onetile);
    cb_pop_front(cb_other, onetile);           // done with this input tile

    // --- Step B: result * gamma (optional) ---
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

    // --- Step C: result + beta (optional) ---
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
}
```

#### Phase 4: Release per-channel parameters (end of channel group)

```cpp
cb_pop_front(cb_bcast, onetile);   // release mean
cb_pop_front(cb_den, onetile);     // release inv_sqrt
if (weight_has_value) cb_pop_front(cb_weight, onetile);
if (bias_has_value)   cb_pop_front(cb_bias, onetile);
```

### Key Compute API Signatures (FPU Variant)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `binary_op_init_common` | `(cb_src_a, cb_src_b, cb_dst)` | One-time HW init for binary ops (unpacker, math, packer) |
| `add_tiles_init_with_dt` | `(icb0, icb1)` | Init FPU add with data format reconfig for fp32 |
| `add_tiles` | `(cb_a, cb_b, tile_idx_a, tile_idx_b, dst_idx)` | FPU: dst[dst_idx] = cb_a[tile_idx_a] + cb_b[tile_idx_b] |
| `sub_tiles_init` | `(icb0, icb1)` | Init FPU subtract |
| `sub_tiles` | `(cb_a, cb_b, tile_idx_a, tile_idx_b, dst_idx)` | FPU: dst[dst_idx] = cb_a[tile_idx_a] - cb_b[tile_idx_b] |
| `mul_tiles_init_with_dt` | `(icb0, icb1)` | Init FPU multiply with data format reconfig |
| `mul_tiles` | `(cb_a, cb_b, tile_idx_a, tile_idx_b, dst_idx)` | FPU: dst[dst_idx] = cb_a[tile_idx_a] * cb_b[tile_idx_b] |
| `rsqrt_tile_init` | `()` | Init SFPU rsqrt |
| `rsqrt_tile` | `(dst_idx)` | SFPU: dst[dst_idx] = 1/sqrt(dst[dst_idx]) (in-place on dst) |
| `binary_dest_reuse_tiles_init` | `<EltwiseBinaryType, EltwiseBinaryReuseDestType>(cb_src)` | Init binary op with dest register reuse |
| `binary_dest_reuse_tiles` | `<EltwiseBinaryType, EltwiseBinaryReuseDestType>(cb_src, tile_idx, dst_idx)` | Binary op: dst[dst_idx] = dst[dst_idx] OP cb_src[tile_idx] |
| `pack_tile_with_dt` | `(dst_idx, cb_out)` | Pack dst register to CB (with fp32 data format reconfig) |
| `tile_regs_acquire` | `()` | Acquire dst registers for math+unpack |
| `tile_regs_commit` | `()` | Transfer dst ownership from math to packer |
| `tile_regs_wait` | `()` | Wait for packer to be ready |
| `tile_regs_release` | `()` | Release dst registers after packing |

### binary_dest_reuse_tiles Pattern (Critical for layer_norm_rm)

This is the key optimization in the FPU variant. Instead of:
1. Compute `input - mean` -> pack to intermediate CB
2. Read intermediate CB + read inv_sqrt CB -> multiply -> pack to output

It does:
1. Compute `input - mean` -> result stays in dst register
2. `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(cb_den, 0, 0)` -- moves dst to SRCA, reads cb_den tile 0 to SRCB, multiplies, result back to dst

This **eliminates one CB round-trip** (write to L1 + read from L1). The result of the subtraction stays in the dst register and is used directly as an operand for the multiplication. The `DEST_TO_SRCA` flag means the current dst value is loaded into SRCA before the binary op executes.

---

## SFPU Kernel Variant Differences

File: `batch_norm_sfpu_kernel.cpp`

Selected when `fp32_dest_acc_en = true`. Key differences:

1. **Initialization**: Uses `unary_op_init_common(cb_other, cb_output_0)` instead of `binary_op_init_common`.

2. **Explicit copy_tile**: All operands must be explicitly unpacked to dst registers using `copy_tile`. FPU operations (add_tiles, sub_tiles) work directly on CBs; SFPU operations (add_binary_tile, sub_binary_tile, mul_binary_tile) work on dst register indices.

3. **Two dst registers used**: Operations use dst slots `i*2` and `i*2+1` for the two operands, then combine.

4. **Example (Phase 1 equivalent)**:
```cpp
tile_regs_acquire();
tile_regs_wait();
copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
copy_tile(cb_batch_var, 0, 0);    // dst[0] = batch_var
add_binary_tile_init();
copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
copy_tile(cb_eps, 0, 1);          // dst[1] = eps
add_binary_tile(0, 1, 0);         // dst[0] = dst[0] + dst[1]
rsqrt_tile_init();
rsqrt_tile(0);                    // dst[0] = 1/sqrt(dst[0])
pack_tile(0, cb_den);
tile_regs_commit();
tile_regs_release();
```

5. **No binary_dest_reuse_tiles**: The SFPU variant uses explicit copy_tile for both operands and separate binary ops. No dest reuse optimization.

6. **Different tile_regs ordering**: Note `tile_regs_acquire(); tile_regs_wait();` immediately together, then operations, then `tile_regs_commit(); tile_regs_release();`. This is a simpler (non-overlapping) pipeline pattern.

---

## Scalar/Constant CB Setup (Epsilon)

The reader kernel fills the epsilon CB once at program start:

```cpp
// Reader kernel (reader_batch_norm.cpp)
union { float f; uint32_t u; } scalar;
scalar.u = eps;  // eps passed as packed uint32_t runtime arg
cb_reserve_back(cb_id_eps, onetile);
#ifdef FILL_WITH_VALUE_FLOAT
    FILL_WITH_VALUE_FLOAT(cb_id_eps, scalar.f);    // float32: fill 1024 elements
#endif
#ifdef FILL_WITH_VALUE
    FILL_WITH_VALUE(cb_id_eps, eps);                // bfloat16: fill 512 uint32_t (packed pairs)
#endif
cb_push_back(cb_id_eps, onetile);
```

The host packs epsilon appropriately:
```cpp
// Program factory
const auto packed_scalar_eps = input_tensor.dtype() == DataType::FLOAT32
    ? std::bit_cast<uint32_t>(scalar)
    : pack_two_bfloat16_into_uint32({scalar, scalar});
```

**Key lesson for layer_norm_rm**: To fill a tile with a scalar value, use `fill_with_val` (for bfloat16/float32). The tile is filled element-by-element in L1 before being pushed to the CB. For bfloat16, elements are packed two-per-uint32.

---

## Per-Channel Parameter Broadcasting (FILL_TILE_WITH_FIRST_ELEMENT)

The writer kernel reads a tile from DRAM for each per-channel parameter, then broadcasts the first element across the entire tile:

```cpp
// Writer kernel (writer_batch_norm.cpp)
cb_reserve_back(cb_id_src, onetile);
uint32_t l1_write_addr = get_write_ptr(cb_id_src);
noc_async_read_tile(tile_offset, src, l1_write_addr);
noc_async_read_barrier();
FILL_TILE_WITH_FIRST_ELEMENT(cb_id_src);  // broadcast element[0] to all 1024 positions
cb_push_back(cb_id_src, onetile);
```

`fill_tile_with_first_element_bfloat16` (from `fill_tile_utils.hpp`):
- Reads the first uint16_t from the tile
- Packs it into a uint32_t (doubled)
- Writes it to all 512 uint32_t positions (covering 1024 bfloat16 elements)

This is needed because the per-channel parameters (mean, var, gamma, beta) are stored as [1, C, 1, 1] tiles where only element [0,0] has the meaningful value. The broadcast ensures every element in the tile has the same value, so standard tile-level elementwise operations produce correct results without needing broadcast-mode instructions.

**Relevance for layer_norm_rm**: If layer_norm_rm computes its own mean/variance via reduction, the results will already be in tile form. However, if the reduction produces a scalar-per-row result, a similar broadcast pattern may be needed. For gamma/beta parameters in layer norm (which are per-feature, i.e., per-column), the broadcast pattern would differ -- potentially using `fill_tile_with_first_row` instead.

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (but linearized row-major) |
| **Grid dimensions** | `compute_with_storage_grid_size.x` x `compute_with_storage_grid_size.y` |
| **Total cores** | Up to all available compute cores |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two groups via `split_work_to_cores`: group_1 gets `ceil(total/cores)` tiles, group_2 gets `floor(total/cores)` tiles |
| **Remainder handling** | Cores not in group_1 or group_2 get 0 tiles and early-exit |

Work is split by total output tiles. The `split_work_to_cores` utility creates two core groups for even distribution. Inactive cores receive zero-filled runtime args and the compute kernel returns immediately (`if (num_tiles == 0) return;`).

---

## Arguments

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | weight_has_value | uint32_t (bool) | Whether gamma is present (1=yes, 0=no) |
| 1 | bias_has_value | uint32_t (bool) | Whether beta is present (1=yes, 0=no) |
| 2 | cb_input | uint32_t | CB index for input tensor (c_0) |
| 3 | cb_batch_mean | uint32_t | CB index for batch mean (c_1) |
| 4 | cb_output_0 | uint32_t | CB index for output (c_2) |
| 5 | cb_batch_var | uint32_t | CB index for batch variance (c_3) |
| 6 | cb_eps | uint32_t | CB index for epsilon (c_4) |
| 7 | cb_den | uint32_t | CB index for intermediate inv_sqrt (c_7) |
| 8 | cb_weight | uint32_t | CB index for gamma (c_5) |
| 9 | cb_tmp_1 | uint32_t | CB index for intermediate scratch (c_8) |
| 10 | cb_bias | uint32_t | CB index for beta (c_6) |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Total tiles to process on this core |
| 1 | freq | uint32_t | Tiles per channel group (Ht * Wt) |
| 2 | counter | uint32_t | Starting tile offset within first channel group (`start_tile_id % cHtWt`) |

### Runtime Arguments (Reader Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar_eps | uint32_t | Epsilon value (packed as float32 or dual bfloat16) |
| 1 | src_addr | uint32_t | Input tensor buffer address |
| 2 | start_tile_id | uint32_t | First tile index for this core |
| 3 | num_tiles | uint32_t | Total tiles to process |
| 4 | cHtWt | uint32_t | Tiles per channel (Ht * Wt) |
| 5 | n_stride | uint32_t | Tile stride for batch dimension |
| 6 | c_stride | uint32_t | Tile stride for channel dimension |
| 7 | N | uint32_t | Number of batches |
| 8 | C | uint32_t | Number of channels |
| 9 | Ht | uint32_t | Height in tiles |
| 10 | Wt | uint32_t | Width in tiles |

### Runtime Arguments (Writer Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | batch_mean_addr | uint32_t | Batch mean buffer address |
| 1 | batch_var_addr | uint32_t | Batch variance buffer address |
| 2 | weight_addr | uint32_t | Gamma buffer address (0 if absent) |
| 3 | bias_addr | uint32_t | Beta buffer address (0 if absent) |
| 4 | dst_addr | uint32_t | Output buffer address |
| 5 | start_tile_id | uint32_t | First tile index for this core |
| 6 | num_tiles | uint32_t | Total tiles to process |
| 7 | HtWt | uint32_t | Tiles per channel |
| 8 | n_stride | uint32_t | Tile stride for batch dimension |
| 9 | c_stride | uint32_t | Tile stride for channel dimension |
| 10 | N | uint32_t | Number of batches |
| 11 | C | uint32_t | Number of channels |
| 12 | Ht | uint32_t | Height in tiles |
| 13 | Wt | uint32_t | Width in tiles |

---

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_batch_norm | RISCV_0 | NOC0 | DRAM (input) | cb_input (c_0), cb_eps (c_4) | Read input tiles via TensorAccessor; fill eps tile with scalar |
| writer_batch_norm | RISCV_1 | NOC1 | DRAM (mean, var, gamma, beta) | cb_batch_mean (c_1), cb_batch_var (c_3), cb_weight (c_5), cb_bias (c_6); reads cb_output (c_2) to DRAM | Read per-channel params, broadcast-fill; write output tiles |
| batch_norm_kernel | RISCV_2 (compute) | N/A | c_0, c_1, c_3, c_4, c_5, c_6 | c_2 (via c_7, c_8 intermediates) | sub, rsqrt, mul (dest_reuse), optional gamma mul, optional beta add |

### Reader Key Logic
- Fills epsilon into cb_eps once at startup (program-lifetime)
- Iterates over tiles in N-C-Ht-Wt order using computed tile offsets
- Uses TensorAccessor for DRAM reads

### Writer Key Logic
- Reads per-channel parameters at the START of each channel group (before the inner tile loop)
- Applies `FILL_TILE_WITH_FIRST_ELEMENT` to broadcast scalar value across full tile
- Writes output tiles in the inner loop
- Uses TensorAccessor for all DRAM reads and writes

### Compute Key Logic
- Two-level loop: outer per-channel-group, inner per-spatial-tile
- Phase 1: inv_sqrt computed once per channel group
- Phase 2: Per-tile normalize with dest register reuse optimization
- Phase 3: Optional gamma/beta affine transform (compile-time conditional)

---

## Pipeline Pattern Summary

All CBs are allocated with capacity=2 tiles and block_size=1 tile, indicating **double-buffering** throughout. This allows the reader/writer to work on the next tile while compute processes the current tile, providing overlap between data movement and computation.

The compute kernel itself does NOT overlap phases within a tile -- each phase is sequential: subtract, multiply-by-invstd, optionally multiply-by-gamma, optionally add-beta. Each phase uses the `tile_regs_acquire/commit/wait/release` protocol, with the exception of the sub+mul fusion via `binary_dest_reuse_tiles` which chains two operations in a single acquire/commit block.

---

## Implementation Notes

### FP32 Dest Accumulation Mode
- When `fp32_dest_acc_en` is true, a completely different compute kernel is compiled (`batch_norm_sfpu_kernel.cpp`)
- All CBs used by compute are configured with `UnpackToDestMode::UnpackToDestFp32` to preserve 32-bit precision during unpacking
- The SFPU variant uses `copy_tile` + `{op}_binary_tile` instead of direct `{op}_tiles` FPU calls

### Dataflow Defines for Data Type
The program factory sets preprocessor defines based on input dtype:
- FLOAT32: `FILL_TILE_WITH_FIRST_ELEMENT = fill_tile_with_first_element<float>`, `FILL_WITH_VALUE_FLOAT = fill_with_val<1024, float>`
- BFLOAT16: `FILL_TILE_WITH_FIRST_ELEMENT = fill_tile_with_first_element_bfloat16`, `FILL_WITH_VALUE = fill_with_val_bfloat16`

### Tile Offset Computation
The reader and writer compute tile offsets using stride-based indexing:
```cpp
tile_offset = start_n * n_stride + start_c * c_stride + start_t;
```
Where strides account for potential padding (via padded_shape). This allows proper traversal of the 4D tensor in tile space.

### Binary Dest Reuse Optimization
The `binary_dest_reuse_tiles` pattern is critical: it fuses `(input - mean) * inv_sqrt` into a single acquire/commit block by keeping the subtraction result in the dst register. This saves one L1 write + read cycle per tile. For layer_norm_rm, a similar pattern could fuse `(x - mean) * inv_sqrt`.

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "What do tile_regs_acquire, tile_regs_commit, tile_regs_wait, tile_regs_release do? How do they control the dest register file pipeline?"
   **Reason**: Understanding the fundamental compute pipeline synchronization protocol.
   **Key Findings**: These four functions form a producer-consumer protocol for the shared Dst register file. acquire() gives ownership to math/unpack. commit() transfers to packer. wait() blocks until packer is ready. release() frees registers for next iteration. Double-buffering of dst registers is controlled by dst_full_sync_en.

2. **Query**: "What is binary_dest_reuse_tiles and what does DEST_TO_SRCA mean?"
   **Reason**: Understanding the key optimization in the FPU compute kernel.
   **Key Findings**: binary_dest_reuse_tiles performs a binary op where one operand comes from the Dst register (not a CB). DEST_TO_SRCA moves the current Dst value to SRCA before the operation. This enables chaining operations without intermediate CB writes, critical for performance.

3. **Query**: "What is the difference between FPU and SFPU compute kernel variants?"
   **Reason**: Understanding why two compute kernel files exist and when each is used.
   **Key Findings**: FPU operations (add_tiles, sub_tiles, mul_tiles) work directly on CBs. SFPU operations (add_binary_tile, etc.) require explicit copy_tile to dst registers first. SFPU is used when fp32_dest_acc_en=true because it provides better 32-bit precision support. FPU max precision is TF32 (19 bits), SFPU supports full 32-bit.

4. **Query**: "What does binary_op_init_common do?"
   **Reason**: Understanding the one-time initialization call in the compute kernel.
   **Key Findings**: Configures three HW components: unpacker (llk_unpack_hw_configure, llk_unpack_AB_init), math unit (llk_math_pack_sync_init, llk_math_hw_configure), packer (llk_pack_hw_configure, llk_pack_init, llk_pack_dest_init). Called once at kernel start.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`
   **Reason**: Understanding the `_with_dt` wrapper functions and helper utilities.
   **Key Information**: `pack_tile_with_dt` adds `pack_reconfig_data_format(icb)` before `pack_tile()` when `FP32_DEST_ACC_EN` is defined. `add_tiles_init_with_dt` adds `reconfig_data_format(icb0, icb1)`. Also provides composite helpers like `mul_tiles_to_cb`, `add_tiles_to_cb`, `sub_tiles_to_cb` that handle the full CB wait/acquire/op/commit/wait/pack/release/pop/push cycle.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp`
   **Reason**: Understanding how scalar values and per-channel parameters are broadcast into full tiles.
   **Key Information**: `fill_with_val_bfloat16(cb_id, packed_scalar)` fills 512 uint32_t with packed bfloat16 pairs. `fill_tile_with_first_element_bfloat16(cb_id)` reads element[0], packs it, fills entire tile. Also provides `fill_tile_with_first_row` and `fill_tile_with_first_column` variants for different broadcast patterns.

3. **Source**: `ttnn/cpp/ttnn/operations/cb_utils.hpp`
   **Reason**: Understanding the `create_cb` utility function used in the program factory.
   **Key Information**: `create_cb(cb_index, program, core_spec, page_size, num_pages, data_format)` creates a CircularBufferConfig with `num_pages * page_size` total size, sets page_size per CB, returns `(cb_index, handle)` tuple.
