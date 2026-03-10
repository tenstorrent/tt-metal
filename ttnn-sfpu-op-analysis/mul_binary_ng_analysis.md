# MUL (binary_ng) Implementation Analysis

## Overview

The MUL operation in the `binary_ng` framework performs element-wise multiplication of two tensors: `c = a * b`. When operating through the SFPU path (as opposed to the FPU path), it uses the `mul_binary_tile` SFPU kernel which computes floating-point multiplication with special handling for bfloat16 rounding and zero-operand semantics. The `binary_ng` program factory is a unified, highly parameterized factory that supports all binary operations (ADD, SUB, MUL, DIV, etc.) through compile-time defines and kernel selection.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

For SFPU MUL specifically:
- `BINARY_SFPU_INIT` is defined as `mul_binary_tile_init();`
- `BINARY_SFPU_OP` is defined as `mul_binary_tile`
- The SFPU path is selected when `operation_attributes.is_sfpu == true`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `c.physical_volume() / tile_hw` (total output tiles) |
| **Loop structure** | Single flat loop over `num_tiles` per core; each iteration processes 1 tile through the read-compute-write pipeline |

The program factory produces 1 output tile per read-compute-write cycle (`num_tiles_per_cycle = 1`). For each tile, both input operands are loaded into DEST registers, the SFPU MUL is applied, and the result is packed out.

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A | Input Tensor B |
|----------|---------------|----------------|
| **Logical shape** | Up to rank-6+ (nD, D, N, C, H, W) | Up to rank-6+ (nD, D, N, C, H, W) |
| **Dimension convention** | Last 5 dims: D, N, C, Ht, Wt (padded, tile-aligned) | Same convention; can differ for broadcast |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED (HEIGHT, WIDTH, BLOCK) | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | Same as A for SFPU MUL (inferred) |

When B is not provided (scalar path), the writer kernel fills a single tile with the packed scalar value.

### Output Tensor

| Property | Output Tensor C |
|----------|----------------|
| **Logical shape** | Broadcasted output shape of A and B |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Matches operation configuration; can differ from input |

### Layout Transformations

No explicit tilize/untilize occurs within this program factory. All tensors must already be in TILE_LAYOUT. Broadcasting is handled by the reader kernel through stride-based index calculations (stride = 0 when a dimension is broadcast). When sharded inputs have different shard shapes than the output, `adjust_to_shape` computes adjusted shard specs.

## Data Flow Pattern

### Two-Tensor Path (both A and B are tensors)

1. **Reader kernel** reads tiles from A into `CB_c_0` and from B into `CB_c_1`, one tile at a time per operand, iterating through the 6D index space (nD, D, N, C, Ht, Wt). Strides handle broadcasting: if a dimension has size 1 in the input, its stride is 0.
2. **Compute kernel** waits for 1 tile in `CB_c_0` (LHS) and 1 tile in `CB_c_1` (RHS), acquires DEST registers, copies both tiles into DEST via `copy_tile`, executes `mul_binary_tile(0, 1, 0)` on the SFPU, packs the result into `CB_c_2`.
3. **Writer kernel** waits for 1 tile in `CB_c_2`, writes it to the output buffer via NoC, then pops.

### Scalar Path (B is a scalar)

1. **Reader kernel** reads tiles from A into `CB_c_0`.
2. **Writer kernel** fills a single tile in `CB_c_1` with the packed scalar (done once), then writes output tiles from `CB_c_2`.
3. **Compute kernel** processes tiles as above but `CB_c_1` always contains the same scalar tile.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_pre_lhs | Input A staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_pre_rhs | Input B staging | 2 tiles (interleaved, tensor) / 1 tile (scalar) / shard volume (sharded) | 1 tile | Double / Single | Reader or Writer (scalar) | Compute | Program |
| c_2 | cb_out | Output staging | 2 tiles (interleaved) or shard volume (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_post_lhs | LHS activation intermediate | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |
| c_4 | cb_post_rhs | RHS activation intermediate | 1 tile | 1 tile | Single | Compute (preprocess) | Compute (main) | Block |

**Notes**:
- CB c_3 and c_4 are only created when LHS or RHS activations are non-empty (i.e., when preprocessing is needed). For plain MUL without activations, they are not allocated; `cb_post_lhs` aliases `cb_pre_lhs` and `cb_post_rhs` aliases `cb_pre_rhs`.
- For sharded tensors, the CB capacity equals the shard volume in tiles, and the buffer is backed directly by the sharded L1 allocation.

## Pipeline Pattern Summary

- **Interleaved path**: Double-buffered on c_0, c_1, c_2. The reader can push the next tile while compute processes the current one. However, with `noc_async_read_barrier()` after each tile read, effective overlap is limited.
- **Sharded path**: All shard tiles are pushed at once; no streaming overlap needed since data is already in L1.

## Index Calculations

The reader kernel uses a 6-dimensional index decomposition based on the output tile ID:

```
tiles_per_n = C * Ht * Wt
tiles_per_d = N * tiles_per_n
tiles_per_nd = D * tiles_per_d

For a given start_tile_id:
  start_nd = start_tile_id / tiles_per_nd
  start_d  = (start_tile_id % tiles_per_nd) / tiles_per_d
  start_n  = ... / tiles_per_n
  start_c  = ... / HtWt
  start_th = ... / Wt
  start_tw = ... % Wt
```

For input A, a `tile_offset` is computed using per-dimension strides:
```
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt
```

Where strides are set to 0 when the corresponding dimension in the input has size 1 (broadcast). The same logic applies to input B with separate strides (`nD_stride_b`, `d_stride_b`, etc.).

TensorAccessor is used for physical memory address translation from logical tile indices.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads within the innermost dimensions (tw loop), with stride-based jumps at dimension boundaries. Each tile is read individually with `noc_async_read_page` followed by `noc_async_read_barrier`, resulting in synchronous reads.
- **Sharded**: All tiles in the shard are published at once via `cb_reserve_back`/`cb_push_back` (data is already in L1).

### Write Pattern
- **Interleaved**: Sequential tile writes, one at a time, with `noc_async_write_page` followed by `noc_async_write_barrier`.
- **Sharded**: Output CB is backed by sharded L1; no explicit writes needed.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (preferred) or arbitrary CoreRangeSet |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `compute_with_storage_grid.x * compute_with_storage_grid.y` (zero-start) or `all_device_cores.num_cores()` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(total_tiles / num_cores)`, group 2 gets `floor(total_tiles / num_cores)` |

The factory uses `split_work_to_cores` which divides total output tiles across cores. If the total tiles do not divide evenly, some cores (core_group_1) process one more tile than others (core_group_2). Cores not in either group receive zero-argument runtime args (no-op).

For sharded inputs, the core grid comes from the shard spec, and each core processes exactly its shard's tiles.

## Arguments

### Compile-Time Arguments

**Compute Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1; tiles produced per read-compute-write cycle |

The operation type is conveyed via preprocessor defines rather than compile-time args:
- `BINARY_SFPU_INIT` = `mul_binary_tile_init();`
- `BINARY_SFPU_OP` = `mul_binary_tile`
- `BCAST_INPUT` = `""` (no broadcast) or `"0"`/`"1"` (broadcast LHS/RHS)

**Reader Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs(A) | uint32_t[] | Compile-time portion of TensorAccessor for input A |
| N+1..M | TensorAccessorArgs(B) | uint32_t[] | Compile-time portion of TensorAccessor for input B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

**Writer Kernel:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs(C) | uint32_t[] | Compile-time portion of TensorAccessor for output C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding is active, 0 otherwise |

### Runtime Arguments

**Compute Kernel (per core):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Number of output tiles this core processes |
| 1 | freq | uint32_t | Broadcast frequency (1 for no-bcast, Ht*Wt for scalar, Wt for col) |
| 2 | counter | uint32_t | Initial counter offset for broadcast tracking |
| 3 | compute_scalar_value | uint32_t | Unused for standard MUL (0); used by quantization ops |

**Reader Kernel (per core, two-tensor path):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Buffer address of input A |
| 1 | c_start_id | uint32_t | Starting output tile ID for this core |
| 2 | a_num_tiles | uint32_t | Number of A shard tiles (0 if interleaved) |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_current_shard_width | uint32_t | Shard width in tiles (0 if interleaved) |
| 5 | nD_stride | uint32_t | A stride for collapsed dims > 5 |
| 6 | d_stride | uint32_t | A stride for D dimension |
| 7 | n_stride | uint32_t | A stride for N dimension |
| 8 | c_stride | uint32_t | A stride for C dimension |
| 9-14 | D, N, C, Ht, Wt, cND | uint32_t | Output shape dimensions |
| 15 | src_addr_b | uint32_t | Buffer address of input B |
| 16-19 | nD_stride_b, d_stride_b, n_stride_b, c_stride_b | uint32_t | B strides for broadcasting |
| 20 | b_num_tiles | uint32_t | Number of B shard tiles (0 if interleaved) |

**Writer Kernel (per core, two-tensor path):**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Buffer address of output C |
| 1 | c_start_id | uint32_t | Starting output tile ID |
| 2 | c_num_tiles | uint32_t | Number of output tiles |
| 3 | c_current_shard_width | uint32_t | Shard width in tiles |
| 4-9 | D, N, C, Ht, Wt, cND | uint32_t | Output shape dimensions |
| 10 | (unused) | uint32_t | Reserved (0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 (A, B) | CB_c_0, CB_c_1 | Read A tiles, read B tiles with broadcast stride logic |
| compute | RISCV_2 | N/A | CB_c_0, CB_c_1 | CB_c_2 | copy_tile to DEST, SFPU mul_binary_tile, pack_tile |
| writer | RISCV_1 | NOC1 | CB_c_2 | DRAM/L1 (C) | Write output tiles |

### Reader Kernel (Two-Tensor Path)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads A into CB c_0 and B into CB c_1 with independent stride calculations. Broadcasting is achieved by setting per-dimension strides to 0 for broadcast dimensions. Uses `TensorAccessor` for physical address mapping. For sharded inputs, simply does `cb_reserve_back`/`cb_push_back` to make existing L1 data visible to compute.

### Reader Kernel (Scalar Path)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Only reads A into CB c_0. B is handled by the writer kernel which fills a tile with the scalar value.

### Writer Kernel (Two-Tensor Path)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Reads output tiles from CB c_2 and writes them to the output buffer. Uses the same 6D index decomposition as the reader for proper tile placement.

### Writer Kernel (Scalar Path)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: First fills a single tile in CB c_1 with the packed scalar value (using `fill_with_val` or `fill_with_val_bfloat16`), then writes output tiles from CB c_2.

### Compute Kernel

This section covers the SFPU no-broadcast compute kernel, which is the primary path for MUL when both inputs have the same shape.

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"        // provides mul_binary_tile, mul_binary_tile_init
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"     // BCAST_OP, HAS_ACTIVATIONS macros
#include "eltwise_utils_sfpu.hpp"       // PREPROCESS macro for activation pre-processing

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);   // runtime arg: how many tiles this core processes

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // always 1

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;    // input A circular buffer
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;    // input B circular buffer
    constexpr auto cb_out = tt::CBIndex::c_2;         // output circular buffer

    // If LHS/RHS activations exist, route through intermediate CBs c_3/c_4; otherwise alias to pre CBs
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);  // initializes unpack/pack pipeline for the given CB pair
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // configure packer for fused ReLU if defined
#endif

    // If no activations at all, init SFPU once outside the loop for efficiency
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT    // expands to: mul_binary_tile_init();
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS: if LHS activations exist, copy tile from cb_pre_lhs -> apply activation -> cb_post_lhs
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);  // wait for 1 LHS tile to be available

        // PREPROCESS: same for RHS
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);  // wait for 1 RHS tile to be available

        cb_reserve_back(cb_out, num_tiles_per_cycle);  // reserve space in output CB

        // If activations present but no post-processing, reinit SFPU each iteration
        // (needed because preprocess may have changed SFPU state)
#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT    // mul_binary_tile_init();
#endif
        tile_regs_acquire();  // acquire exclusive access to DEST registers

        // Copy LHS tile from CB to DEST register 0 (even slot: i*2 = 0)
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);  // configure unpacker for LHS dtype
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);  // copy tile i from cb_post_lhs to DEST[0]
        }

        // Copy RHS tile from CB to DEST register 1 (odd slot: i*2+1 = 1)
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);  // configure unpacker for RHS dtype
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);  // copy tile i from cb_post_rhs to DEST[1]

            // If post-activations exist, reinit SFPU each tile (to restore state after post-processing)
#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT    // mul_binary_tile_init();
#endif
            // Execute the SFPU binary operation: DEST[0] = DEST[0] * DEST[1]
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);  // expands to: mul_binary_tile(0, 1, 0)

            // Apply any post-processing activations (e.g., typecast) to DEST[0]
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();  // signal that DEST register writes are complete

        tile_regs_wait();    // wait for DEST data to be ready for packing

        // Pack result from DEST[0] into the output CB
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);  // pack DEST[0] to cb_out
        }
        tile_regs_release();  // release DEST registers

        cb_push_back(cb_out, num_tiles_per_cycle);      // publish output tile to writer
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);  // free consumed LHS tile
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);  // free consumed RHS tile
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`

(Identical implementation exists at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE).
// Implements the "add 0x7fff + LSB" algorithm for correct tie-breaking behavior.
// This is needed because the SFPU operates in float32 internally, but the FPU
// (matrix engine) uses bfloat16 — so the SFPU must match FPU rounding to ensure
// numerical consistency.
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Reinterpret the float32 value as an unsigned integer for bit manipulation
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract bit 16 (the LSB of the future bf16 mantissa) for tie-breaking
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb. This implements banker's rounding:
    // - lower 16 bits > 0x8000: carry propagates -> rounds up
    // - lower 16 bits < 0x8000: no carry -> rounds down (truncation)
    // - lower 16 bits == 0x8000 (exact tie) and lsb==0: 0x7fff+0=0xffff, no carry -> stays even (round down)
    // - lower 16 bits == 0x8000 (exact tie) and lsb==1: 0x7fff+1=0x8000, carry -> rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Mask off the lower 16 bits to yield a bf16 value stored in the upper 16 bits of a float32
    bits = bits & 0xFFFF0000U;

    // Reinterpret back to float — this is now a valid float32 whose value equals the bf16 representation
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

// The MUL-specific SFPU binary kernel function.
// Template parameters:
//   APPROXIMATION_MODE: not used by MUL (passed through for API uniformity)
//   BINOP: BinaryOp::MUL
//   ITERATIONS: 8 — processes 8 rows of 4 elements each per call (one 16x16 face)
//   is_fp32_dest_acc_en: whether DEST accumulator is in fp32 mode
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Each tile in DEST occupies 32 rows in SFPI addressing (64 / SFP_DESTREG_STRIDE = 32).
    // A 32x32 tile is laid out as 4 faces of 16x16. Each face has 16 rows, but SFPI addresses
    // them as 8 "rows" of 4-wide SIMD vectors (8 iterations * 4 lanes = 32 elements per face half,
    // covering 8 logical rows of 4 columns).
    constexpr uint dst_tile_size_sfpi = 32;

    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row (4 elements) from input tile 0 at the current SFPU row pointer position.
        // dst_index_in0 * dst_tile_size_sfpi computes the base offset for tile 0 in DEST.
        // The dst_reg++ at the end advances the implicit row pointer, so successive iterations
        // read successive rows.
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

        // Load one row (4 elements) from input tile 1
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Element-wise floating-point multiply on the SFPU's 4-wide SIMD datapath.
        // The SFPU performs this as a native float32 multiply instruction.
        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // When output is bfloat16, apply software Round-to-Nearest-Even to match FPU behavior.
            // The SFPU naturally produces float32 results; this truncates to bf16 precision.
            result = float32_to_bf16_rne(result);

            // Special case: to match FPU semantics, 0 * x = 0 and x * 0 = 0.
            // Without this, 0 * Inf would produce NaN on the SFPU (IEEE standard),
            // but the FPU produces 0 for bfloat16. This conditional ensures consistency.
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        // Write the result back to the output tile's DEST register at the current row
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the implicit SFPU row pointer to the next row across all tiles
        sfpi::dst_reg++;
    }
}

// Initialization function for SFPU binary operations.
// Called once (or per-iteration when activations change SFPU state) to configure
// the SFPU pipeline for the specific binary operation type.
template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `sfpi::dst_reg[offset]` (load) | Loads a 4-element vector from DEST register file at the given row offset |
| `sfpi::dst_reg[offset] = val` (store) | Stores a 4-element vector to DEST register file at the given row offset |
| `sfpi::dst_reg++` | Advances the implicit SFPU row pointer by 1 row (4 elements) |
| `in0 * in1` (vFloat multiply) | 4-wide SIMD floating-point multiplication on the SFPU datapath |
| `sfpi::reinterpret<vUInt>(val)` | Bitwise reinterpret of vFloat as vUInt (no data movement) |
| `sfpi::reinterpret<vFloat>(val)` | Bitwise reinterpret of vUInt as vFloat |
| `bits >> 16` | 4-wide unsigned right shift by 16 bits |
| `bits & mask` | 4-wide bitwise AND |
| `bits + val` | 4-wide unsigned integer addition |
| `v_if(cond) { ... } v_endif` | Predicated execution: sets per-lane condition codes based on comparison, only executes body for lanes where condition is true |
| `in0 == 0` | 4-wide comparison against zero, sets condition codes |

#### SFPU Register Usage

- **DEST registers**: The SFPU reads and writes to the DEST register file. For MUL:
  - `DEST[0]` (at `dst_index_in0 * 32 + row`): Contains the LHS tile data (loaded by `copy_tile` prior to SFPU execution)
  - `DEST[1]` (at `dst_index_in1 * 32 + row`): Contains the RHS tile data
  - `DEST[0]` (at `dst_index_out * 32 + row`): Receives the output (overwrites LHS in-place since `odst == idst0`)
- **SFPU row pointer** (`dst_reg++`): An implicit pointer that advances by 1 row (4 elements) per iteration. Over 8 iterations, it covers 32 elements (half a 16x16 face).
- **vFloat registers**: `in0`, `in1`, `result` are 4-wide SIMD vector registers local to the SFPU
- **vUInt registers**: `bits`, `lsb` used in the `float32_to_bf16_rne` helper for bit manipulation

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `cb_wait_front` on both input CBs, ensuring data is available.
2. **Unpack to DEST**: `copy_tile(cb_post_lhs, 0, 0)` unpacks the LHS tile from CB c_0 into DEST[0]. `copy_tile(cb_post_rhs, 0, 1)` unpacks the RHS tile into DEST[1]. The `copy_tile_to_dst_init_short_with_dt` calls configure the unpacker for the appropriate data type.
3. **SFPU dispatch**: `mul_binary_tile(0, 1, 0)` is called, which invokes `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(0, 1, 0)`.
4. **Face iteration**: The LLK params function `_llk_math_eltwise_binary_sfpu_params_` iterates over all 4 faces of the tile (in RC vector mode). For each face, it calls `calculate_sfpu_binary_mul` with ITERATIONS=8.
5. **Per-face computation**: Each call processes 8 rows of 4 elements (32 elements per half-face). The SFPU loads pairs of elements from DEST[0] and DEST[1], multiplies them, optionally applies bf16 rounding and zero-handling, and writes back to DEST[0].
6. **Face advancement**: Between faces, `TTI_SETRWC` instructions advance the DEST register read/write pointer by 16 rows (the size of one face).
7. **Pack**: After all faces are processed, `pack_tile(0, cb_out)` reads the result from DEST[0] and packs it into the output CB.

#### SFPU Configuration

- **`UnpackToDestMode::UnpackToDestFp32`**: For SFPU MUL (non-POWER), all input CBs have unpack-to-dest set to FP32 mode. This ensures tiles are unpacked as float32 into DEST, regardless of their storage format.
- **`DST_ACCUM_MODE`**: Template parameter reflecting `fp32_dest_acc_en`. When true, DEST holds full fp32; when false, the bf16 rounding path is taken.
- **`APPROX`**: The approximation mode flag (global compile-time constant). Not meaningfully used by the MUL kernel but passed for API uniformity.
- **`BINARY_SFPU_INIT` / `mul_binary_tile_init()`**: Calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()` which initializes SFPU state, configures address modifiers, and resets hardware counters.

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The `calculate_sfpu_binary_mul` implementation is identical between Wormhole B0 and Blackhole architectures. Both use the same float32 multiply, bf16 RNE rounding, and zero-handling logic.
- The `float32_to_bf16_rne` function is a pure SFPI software implementation (not a hardware instruction), present identically on both architectures.
- The `_llk_math_eltwise_binary_sfpu_params_` iteration function (from tt_llk submodule) handles face traversal using `TTI_SETRWC` instructions, which are available on both architectures.

## Implementation Notes

1. **MUL has a dedicated SFPU kernel**: Unlike ADD/SUB which use the generic `calculate_sfpu_binary` function, MUL uses `calculate_sfpu_binary_mul` — a specialized variant that adds bf16 rounding and zero-operand handling to match FPU semantics.

2. **BF16 rounding consistency**: The `float32_to_bf16_rne` function exists solely to ensure that SFPU MUL produces the same results as FPU MUL for bfloat16 data. The SFPU operates in float32 natively, so without explicit rounding, results would have higher precision than FPU results, causing inconsistencies.

3. **Zero-multiply special case**: IEEE 754 specifies that 0 * Inf = NaN, but the Tenstorrent FPU produces 0 for bfloat16 multiplication when either operand is zero. The SFPU kernel explicitly checks for this case to maintain consistency.

4. **Activation fusion**: The binary_ng framework supports fusing pre-processing (LHS/RHS activations) and post-processing (e.g., typecast, ReLU) directly into the compute kernel, avoiding extra kernel launches and memory traffic.

5. **Broadcast handling**: The factory supports 9 broadcast types (NONE, SCALAR_A/B, ROW_A/B, COL_A/B, ROW_A_COL_B, ROW_B_COL_A). Each selects different reader and compute kernels. The no-broadcast case analyzed here is the most common for same-shape MUL.

6. **Scalar optimization**: When B is a scalar, a single tile is filled with the scalar value by the writer kernel, and the compute kernel reads it repeatedly from CB c_1. The CB for scalar input has capacity of only 1 tile.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What kernels does it use, how does it handle broadcasting, and what is the core distribution strategy?"
   **Reason**: Initial orientation on the binary_ng architecture before reading source code.
   **Key Findings**: Confirmed the factory supports multiple broadcast types via kernel selection, uses split_work_to_cores for load balancing, and has distinct kernel variants for SFPU vs FPU paths.

2. **Query**: "What is the binary_ng SFPU compute kernel? How does it differ from FPU-based binary operations? Where are the SFPU kernel files located?"
   **Reason**: Needed to understand the SFPU-specific kernel path and file locations.
   **Key Findings**: SFPU kernels are in `kernels/compute/eltwise_binary_sfpu*.cpp`. SFPU supports more operation types than FPU. The `UnpackToDestFp32` mode is used for SFPU operations. File path is determined by `get_kernel_file_path`.

3. **Query**: "What does _llk_math_eltwise_binary_sfpu_params_ do? How does it iterate over tile faces?"
   **Reason**: This function is the core dispatch mechanism that calls the SFPU computation function per face but lives in the tt_llk submodule (not checked out).
   **Key Findings**: It iterates over 4 faces in RC vector mode, calling the SFPU function for each face and using `TTI_SETRWC` to advance the DEST register pointer between faces. Each call processes 8 rows of 4 elements.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Contains the `OpConfig` constructor and `get_sfpu_init_fn` that map `BinaryOpType::MUL` to `mul_binary_tile_init()` / `mul_binary_tile`.
   **Key Information**: For SFPU MUL, the defines are `BINARY_SFPU_INIT = "mul_binary_tile_init();"` and `BINARY_SFPU_OP = "mul_binary_tile"`. Integer variants use `mul_int_tile` instead.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Contains the ckernel-level API that bridges the compute kernel to the LLK layer.
   **Key Information**: `mul_binary_tile` dispatches to `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>`, which is distinct from the generic `llk_math_eltwise_binary_sfpu_binop` used by ADD/SUB.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`
   **Reason**: Contains the actual SFPU kernel implementation.
   **Key Information**: `calculate_sfpu_binary_mul` performs element-wise float32 multiply with bf16 RNE rounding and zero-operand handling. Identical on Wormhole and Blackhole.
