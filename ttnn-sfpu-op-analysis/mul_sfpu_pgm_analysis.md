# MUL (Element-Wise Multi-Core SFPU) Implementation Analysis

## Overview

The MUL operation performs element-wise multiplication of two input tensors using the SFPU (Special Function Processing Unit) on Tenstorrent hardware. Unlike the FPU-based `mul_tiles` path (which uses the matrix engine), this SFPU path executes multiplication via SFPI vector instructions on the SFPU co-processor. The SFPU path is selected by `get_defines_fp32` when the operation is dispatched through `ElementWiseMultiCoreSfpu`.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

For floating-point types (BFLOAT16, FLOAT32), the defines resolve to:
- `BINOP_INIT` = `mul_binary_tile_init();`
- `BINARY_SFPU_OP` = `mul_binary_tile(i*2, i*2+1, i*2);`

For integer types (INT32, UINT32, UINT16), a separate `mul_int_tile` path is used instead.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | `block_size` tiles (1 in interleaved mode; up to `find_max_block_size(num_tiles_per_shard)` in sharded mode) |
| **Total units** | `num_tiles = a.physical_volume() / TILE_HW` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_size` tiles per block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A (src0) | Input Tensor B (src1) |
|----------|----------------------|----------------------|
| **Logical shape** | Arbitrary (must match B) | Arbitrary (must match A) |
| **Dimension convention** | NHWC | NHWC |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16 / FLOAT32 / INT32 / UINT32 / UINT16 | BFLOAT16 / FLOAT32 / INT32 / UINT32 / UINT16 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as inputs |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as output dtype (may differ from inputs) |

### Layout Transformations

No tilize/untilize operations. When `is_fp32_dest_acc_en` is false (non-FP32 output), the SFPU kernel applies software Round-to-Nearest-Even (RNE) conversion from float32 back to bfloat16 after multiplication. This is because the SFPU always computes in float32 internally.

## Data Flow Pattern

1. **Reader** reads tiles from DRAM/L1 for both inputs (or marks sharded CBs as ready)
2. **Reader** pushes one tile at a time into `cb_in0` (CB c_0) and `cb_in1` (CB c_1)
3. **Compute** waits for `per_core_block_size` tiles in both `cb_inp0` and `cb_inp1`
4. **Compute** copies input A tiles to even DEST registers (i*2) and input B tiles to odd DEST registers (i*2+1)
5. **Compute** executes `mul_binary_tile(i*2, i*2+1, i*2)` which multiplies DEST[i*2] * DEST[i*2+1] and writes result to DEST[i*2]
6. **Compute** packs result from DEST[i*2] into `cb_out0` (CB c_2)
7. **Writer** reads tiles from `cb_out0` and writes them to DRAM/L1

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (reader pushes 1 at a time) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile (reader pushes 1 at a time) | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_out0 | Output staging | 2 * max_block_size tiles (interleaved) or num_tiles_per_shard (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |
| c_3 | cb_interm0 | Pre-scale input A (optional) | max_block_size tiles | max_block_size tiles | Single | Compute (pre-scale) | Compute (main) | Block |
| c_4 | cb_interm1 | Pre-scale input B (optional) | max_block_size tiles | max_block_size tiles | Single | Compute (pre-scale) | Compute (main) | Block |

**Note**: CBs c_3 and c_4 are only created when pre-scaling defines (`SFPU_OP_INIT_PRE_IN0_0`, `SFPU_OP_INIT_PRE_IN1_0`) are present. For a plain MUL operation, these are not used.

## Pipeline Pattern Summary

- **Interleaved mode**: CBs c_0, c_1, and c_2 each have capacity of `2 * max_block_size` tiles with block size of 1 tile, enabling **double-buffering** between reader/compute and compute/writer.
- **Sharded mode**: CBs are backed by the tensor's globally-allocated L1 buffer. The entire shard is available at once; no streaming overlap is needed.

## Index Calculations

- **Interleaved path**: Reader iterates linearly from `start_id` to `start_id + num_tiles`, reading tile `tile_id` from both src0 and src1 via `TensorAccessor`. The accessor maps the linear tile ID to the correct DRAM bank and offset.
- **Block/width sharded path**: Reader iterates over `block_height` rows and `block_width` columns. The start tile ID is calculated as: `start_id = (core_row * block_height * block_width * num_shards_per_width) + (core_col * block_width)`. Row stride between shard rows is `num_cores_y * block_width`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads via `noc_async_read_tile`. Each tile is read from its interleaved bank location. Both inputs are read in lockstep (same tile ID for src0 and src1), with a `noc_async_read_barrier` after each pair.
- **Sharded**: No DRAM reads. The CB is pointed directly at the L1 shard buffer via `set_globally_allocated_address`. Reader just calls `cb_reserve_back` / `cb_push_back` to signal data availability.

### Write Pattern
- **Interleaved**: Sequential tile-by-tile writes via `noc_async_write_page`. Writer iterates from `start_id` for `num_pages` tiles.
- **Sharded**: No DRAM writes. Output CB is backed by the output tensor's L1 buffer. Writer just calls `cb_wait_front` to signal completion.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-major) for interleaved; matches shard grid for sharded |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `num_cores` (from `split_work_to_cores` or shard grid) |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_tiles / num_cores)` tiles, group 2 gets `floor(num_tiles / num_cores)` tiles |

The factory uses `tt::tt_metal::split_work_to_cores` for interleaved tensors, which divides tiles evenly with remainder distributed to `core_group_1`. For sharded tensors, work per core equals the shard size. A `zero_start_grid` optimization is applied when the worker grid is a single rectangle starting at (0,0).

## Arguments

### Compile-Time Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | uint32_t[] | Accessor args for src0 (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | uint32_t[] | Accessor args for src1 (omitted if IN1_SHARDED) |

**Writer kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | uint32_t[] | Accessor args for output buffer |

**Compute kernel**: No explicit compile-time args. Configuration is via preprocessor defines:

| Define | Value (for MUL) | Description |
|--------|-----------------|-------------|
| `BINOP_INIT` | `mul_binary_tile_init();` | SFPU init function for mul |
| `BINARY_SFPU_OP` | `mul_binary_tile(i*2, i*2+1, i*2);` | SFPU mul operation call |
| `fp32_dest_acc_en` | true if output is Float32/Int32/UInt32 | Enables FP32 accumulation in DEST |
| `UnpackToDestMode` | `UnpackToDestFp32` for all CBs (non-POWER ops) | Unpacks data directly to DEST in FP32 |

### Runtime Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Base address of input tensor A |
| 1 | src1_addr | uint32_t | Base address of input tensor B |
| 2 | num_tiles | uint32_t | Total tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

**Compute kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Tiles per block |

**Writer kernel** (interleaved, non-block-sharded):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Base address of output tensor |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output writes |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM/L1 | CB c_0, CB c_1 | Read src0 and src1 tiles |
| compute | MATH (RISCV_2) | N/A | CB c_0, CB c_1 | CB c_2 | copy_tile + mul_binary_tile + pack_tile |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 | Write output tiles |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Handles four modes via compile-time defines: both interleaved, src0 sharded only, src1 sharded only, both sharded. In the interleaved path, reads one tile at a time from each input with a barrier after each pair. In the block/width sharded path, iterates over a 2D tile grid with stride `num_cores_y * block_width`.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Standard single-tile writer. For sharded output, just calls `cb_wait_front` to confirm data is written. For interleaved output, writes one tile at a time via `noc_async_write_page` with `noc_async_writes_flushed` after each tile for flow control.

### Compute Kernel

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"     // provides mul_binary_tile, mul_binary_tile_init
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/xlogy.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// PRE_SCALE is true if either input has a pre-scaling SFPU op (e.g., for DIV, LOGADDEXP)
// For plain MUL, neither SFPU_OP_INIT_PRE_IN0_0 nor SFPU_OP_INIT_PRE_IN1_0 is defined
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime args: how many blocks and how many tiles per block
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);    // number of blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;    // input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;    // input B circular buffer

    // For MUL: no pre-scaling, so cb_inp0 == cb_in0 and cb_inp1 == cb_in1
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;   // pre-scaled input A (not used for MUL)
#else
    constexpr auto cb_inp0 = cb_in0;              // directly use cb_in0 for MUL
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;   // pre-scaled input B (not used for MUL)
#else
    constexpr auto cb_inp1 = cb_in1;              // directly use cb_in1 for MUL
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // output circular buffer

    // Initialize unary op common state (sets up packer, unpacker config)
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    // Not used for plain MUL unless fused with ReLU activation
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // === PRE-SCALING SECTION (skipped for plain MUL) ===
#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        // Would apply pre-scaling to input A (e.g., exp for LOGADDEXP)
        // Not active for MUL
        cb_wait_front(cb_in0, per_core_block_size);
        cb_reserve_back(cb_inp0, per_core_block_size);
        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN0_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, i);
            SFPU_OP_FUNC_PRE_IN0_0
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp0);
        }
        tile_regs_release();
        cb_pop_front(cb_in0, per_core_block_size);
        cb_push_back(cb_inp0, per_core_block_size);
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
        // Would apply pre-scaling to input B (e.g., recip for DIV)
        // Not active for MUL
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_inp1, per_core_block_size);
        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN1_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in1, i, i);
            SFPU_OP_FUNC_PRE_IN1_0
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp1);
        }
        tile_regs_release();
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_inp1, per_core_block_size);
#endif

        // === MAIN COMPUTATION SECTION ===

        // Wait for input tiles to be available in both input CBs
        cb_wait_front(cb_inp0, per_core_block_size);   // blocks until reader has produced block_size tiles in cb_in0
        cb_wait_front(cb_inp1, per_core_block_size);   // blocks until reader has produced block_size tiles in cb_in1
        cb_reserve_back(cb_out0, per_core_block_size); // reserve output space

        // Acquire DEST registers for SFPU use
        tile_regs_acquire();
        tile_regs_wait();

        // Initialize copy_tile for unpacking input B data type, targeting input A's CB
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // Copy input A tile i from cb_inp0 to DEST register i*2 (even slots)
            copy_tile(cb_inp0, i, i * 2);
        }

        // Re-initialize copy_tile for input B's data type
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // Copy input B tile i from cb_inp1 to DEST register i*2+1 (odd slots)
            copy_tile(cb_inp1, i, i * 2 + 1);

            // For MUL: BINOP_INIT expands to mul_binary_tile_init()
            // This initializes the SFPU for binary multiplication
#ifdef BINOP_INIT
            BINOP_INIT    // mul_binary_tile_init();
#endif
            // (Other INIT macros for INT ops, bitwise, shift -- not active for float MUL)
#ifdef ADD_INT_INIT
            ADD_INT_INIT
#endif
#ifdef SUB_INT_INIT
            SUB_INT_INIT
#endif
#ifdef MUL_INT_INIT
            MUL_INT_INIT
#endif
#ifdef LT_INT32_INIT
            LT_INT32_INIT
#endif
#ifdef GT_INT32_INIT
            GT_INT32_INIT
#endif
#ifdef GE_INT32_INIT
            GE_INT32_INIT
#endif
#ifdef LE_INT32_INIT
            LE_INT32_INIT
#endif
#ifdef BITWISE_INIT
            BITWISE_INIT
#endif
#ifdef BITWISE_UINT16_INIT
            BITWISE_UINT16_INIT
#endif
#ifdef SHIFT_INIT
            SHIFT_INIT
#endif

            // For MUL: BINARY_SFPU_OP expands to mul_binary_tile(i*2, i*2+1, i*2)
            // This performs element-wise multiplication on the SFPU:
            //   DEST[i*2] = DEST[i*2] * DEST[i*2+1]
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP   // mul_binary_tile(i*2, i*2+1, i*2);
#endif

            // Post-op chain (e.g., fused activations) -- not active for plain MUL
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Pack result from DEST[i*2] to the output circular buffer
            pack_tile(i * 2, cb_out0);
        }

        // Signal DEST registers are done (commit) and release them
        tile_regs_commit();
        tile_regs_release();

        // Release input tiles and push output tiles
        cb_pop_front(cb_inp0, per_core_block_size);    // free input A tiles for reader reuse
        cb_pop_front(cb_inp1, per_core_block_size);    // free input B tiles for reader reuse
        cb_push_back(cb_out0, per_core_block_size);    // signal output tiles ready for writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (identical for Wormhole: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`)

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

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This implements the "add 0x7fff + LSB" algorithm for correct tie-breaking.
// Needed because SFPU computes in float32 but output may be bfloat16.
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Reinterpret the float32 value as an unsigned integer for bit manipulation
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract bit 16 (the LSB of the future bf16 mantissa) for tie-breaking
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add rounding bias: 0x7FFF + lsb
    // This implements banker's rounding (round-to-nearest-even):
    //   lower 16 bits > 0x8000 => always rounds up
    //   lower 16 bits < 0x8000 => always rounds down
    //   lower 16 bits == 0x8000 (exact tie) => rounds to even (lsb decides)
    bits = bits + 0x7fffU + lsb;

    // Truncate lower 16 bits to produce bf16 stored in upper 16 bits of float32
    bits = bits & 0xFFFF0000U;

    // Reinterpret back as float (now effectively a bf16 value in float32 container)
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

// calculate_sfpu_binary_mul: the specialized SFPU kernel for element-wise multiplication
// Template parameters:
//   APPROXIMATION_MODE: not used for mul (no approximation needed)
//   BINOP: BinaryOp::MUL
//   ITERATIONS: 8 (processes 8 rows per call; 4 calls per tile = 32 rows total)
//   is_fp32_dest_acc_en: true if DEST accumulator is in FP32 mode
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Each tile occupies 32 SFPI-addressable rows in DEST (64 rows / SFP_DESTREG_STRIDE=2)
    constexpr uint dst_tile_size_sfpi = 32;

    // Process ITERATIONS=8 rows per invocation
    // The params wrapper calls this function 4 times (once per face), each advancing dst_reg
    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row (32 elements) from input A's tile in DEST
        // dst_index_in0 is the tile index (e.g., i*2), multiplied by 32 to get the row offset
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

        // Load one row (32 elements) from input B's tile in DEST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Perform element-wise multiplication using SFPU's float32 multiply
        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // When output is bfloat16, apply software RNE rounding
            // This ensures the SFPU result matches FPU bfloat16 precision
            result = float32_to_bf16_rne(result);

            // Special case: match FPU behavior where 0 * anything = 0
            // Without this, 0 * inf would produce NaN on SFPU but 0 on FPU
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        // Write result back to the output tile's position in DEST
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the SFPI row pointer by 1 (moves to next row within the face)
        sfpi::dst_reg++;
    }
}

// sfpu_binary_init: initialization for binary SFPU operations
// For MUL, BINOP is BinaryOp::MUL which does not match DIV, POW, or XLOGY,
// so _sfpu_binary_init_ is essentially a no-op (no reciprocal or log init needed)
template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
    // For MUL: _sfpu_binary_init_ does nothing (no special init required)
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `sfpi::dst_reg[offset]` (load) | Reads a vector of 32 float32 elements from DEST register at the given row offset |
| `sfpi::dst_reg[offset]` (store) | Writes a vector of 32 float32 elements to DEST register at the given row offset |
| `sfpi::dst_reg++` | Increments the SFPI DEST row pointer by 1 (advances to next row) |
| `in0 * in1` (vFloat multiply) | SFPU vector float32 multiplication of 32 elements in parallel |
| `sfpi::reinterpret<vUInt>(v)` | Reinterprets float32 vector bits as unsigned integer vector (no data movement) |
| `sfpi::reinterpret<vFloat>(v)` | Reinterprets unsigned integer vector bits as float32 vector |
| `bits >> 16` | Vector right shift by 16 bits |
| `bits & mask` | Vector bitwise AND |
| `bits + scalar` | Vector unsigned integer addition |
| `v_if(cond) { ... } v_endif` | Predicated execution: conditionally executes body based on per-lane condition |
| `TTI_SETRWC` | (In params wrapper) Advances the DEST read/write counter between faces |
| `TTI_STALLWAIT` | (In start function) Stalls until SFPU is available on the math engine |

#### SFPU Register Usage

- **DEST registers**: Two input tiles are loaded into even (i*2) and odd (i*2+1) DEST tile slots. Each tile occupies 32 SFPI-addressable rows. The result overwrites the even slot (i*2).
- **SFPI row pointer** (`dst_reg`): Auto-incremented via `dst_reg++` after each row iteration. Reset between faces via `TTI_SETRWC` in the params wrapper.
- **LREG (local registers)**: Implicitly used by SFPI for intermediate vector values (`in0`, `in1`, `result`, `bits`, `lsb`). The SFPU has a limited set of vector local registers.

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` and `tile_regs_wait()` to gain exclusive access to DEST registers.
2. **Unpack to DEST**: `copy_tile(cb_inp0, i, i*2)` unpacks input A tile `i` from CB c_0 into DEST slot `i*2`. `copy_tile(cb_inp1, i, i*2+1)` unpacks input B tile `i` from CB c_1 into DEST slot `i*2+1`. The `UnpackToDestFp32` mode ensures data arrives in DEST as float32.
3. **SFPU init**: `mul_binary_tile_init()` calls through to `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()`, which calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` (configures SFPU config reg, address modes, resets counters) and then `sfpu_binary_init<APPROX, BinaryOp::MUL>()` (no-op for MUL since no reciprocal or log init is needed).
4. **SFPU dispatch**: `mul_binary_tile(i*2, i*2+1, i*2)` calls `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(i*2, i*2+1, i*2)`.
5. **Params wrapper**: `_llk_math_eltwise_binary_sfpu_params_` performs:
   - `_llk_math_eltwise_binary_sfpu_start_`: sets DEST write address and stalls until SFPU is ready
   - In RC (default) mode: loops over 4 faces, calling `calculate_sfpu_binary_mul` for each face (8 iterations = 8 rows per face), then advances DEST pointer by 16 rows via two `TTI_SETRWC` calls between faces
   - `_llk_math_eltwise_binary_sfpu_done_`: clears DEST register address
6. **SFPU math**: For each of the 8 iterations per face, `calculate_sfpu_binary_mul` loads one row from each input tile, multiplies them, optionally applies bf16 RNE rounding and zero-handling, and writes the result back.
7. **Pack**: `pack_tile(i*2, cb_out0)` packs the result from DEST slot `i*2` into the output CB.
8. **Release**: `tile_regs_commit()` and `tile_regs_release()` free DEST registers.

#### SFPU Configuration

- **`fp32_dest_acc_en`**: Enabled when output dtype is Float32, Int32, or UInt32. Controls whether DEST accumulator operates in 32-bit mode (larger tiles, fewer slots) or 16-bit mode.
- **`UnpackToDestFp32`**: Set for all input CBs (except in POWER mode). This ensures data is unpacked directly into DEST registers in float32 format, bypassing the SRC registers.
- **`APPROX` (APPROXIMATION_MODE)**: Compile-time flag; not meaningfully used by the MUL kernel since multiplication does not require approximation.
- **`DST_ACCUM_MODE`**: Passed as the `is_fp32_dest_acc_en` template parameter. When false, the kernel applies software bf16 RNE rounding after multiplication.
- **Address mode**: ADDR_MOD_7 is configured with zero increments for srca, srcb, and dest (since SFPI manages its own addressing via `dst_reg` pointer).

#### Hardware Compatibility Notes

- The `ckernel_sfpu_binary.h` file is **identical** between Wormhole B0 and Blackhole architectures.
- The `llk_math_eltwise_binary_sfpu_binop.h` LLK wrapper is also **identical** between both architectures.
- The `llk_math_eltwise_binary_sfpu_params.h` (from `tt_llk` submodule) and `llk_math_eltwise_binary_sfpu.h` base infrastructure files may have minor differences in register configuration between architectures, but the SFPU kernel logic itself is architecture-independent.
- The `float32_to_bf16_rne` software rounding function is used on both architectures to ensure consistent bfloat16 results between SFPU and FPU paths.

## Implementation Notes

1. **SFPU vs FPU path**: The MUL operation has two paths -- an FPU path (via `mul_tiles` in the `ElementWiseMultiCore` factory) that uses the matrix engine, and this SFPU path that uses the vector SFPU. The SFPU path is selected when `get_defines_fp32` is used, which happens when the operation is routed through `ElementWiseMultiCoreSfpu`.

2. **bf16 RNE rounding**: When `is_fp32_dest_acc_en` is false, the SFPU kernel explicitly performs software Round-to-Nearest-Even conversion. This is necessary because the SFPU computes in float32 but the output needs bfloat16 precision. The algorithm uses the classic `add 0x7FFF + LSB` technique for correct banker's rounding.

3. **Zero multiplication handling**: The `v_if(in0 == 0 || in1 == 0) { result = 0.0f; }` guard ensures that `0 * inf = 0` and `0 * NaN = 0`, matching the FPU's bfloat16 behavior. Without this, IEEE 754 float32 rules would produce NaN for `0 * inf`.

4. **Tile interleaving in DEST**: Input A occupies even DEST slots (0, 2, 4, ...) and input B occupies odd slots (1, 3, 5, ...). This allows processing multiple tiles per block while keeping both operands resident in DEST. The maximum block size is limited by available DEST slots (4 tiles per operand in 16-bit mode, 2 in 32-bit mode).

5. **Init called per-tile**: The `BINOP_INIT` (`mul_binary_tile_init()`) is called inside the per-tile loop. This re-initializes the SFPU state for each tile, which is necessary because `copy_tile` operations between tiles may alter SFPU configuration.

6. **Sharding support**: The factory supports HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED memory layouts. For sharded inputs, the CB is backed directly by the tensor's L1 buffer (via `set_globally_allocated_address`), eliminating DRAM reads. The block/width sharded path uses a different writer kernel (`writer_unary_sharded_blocks_interleaved_start_id.cpp`) when the output is interleaved.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary SFPU element-wise multi-core program factory work? What kernels does it use (reader, compute, writer)? How does it distribute work across cores for binary operations like mul?"
   **Reason**: Needed to understand the overall architecture and kernel assignment before reading source code.
   **Key Findings**: Confirmed three-kernel pipeline (reader/compute/writer), SPMD work distribution via `split_work_to_cores`, and that `ElementWiseMultiCoreSfpu` is selected for same-shape SFPU-supported operations.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Needed to understand how `get_defines_fp32` maps `BinaryOpType::MUL` to specific preprocessor defines.
   **Key Information**: MUL with float types produces `BINOP_INIT = mul_binary_tile_init()` and `BINARY_SFPU_OP = mul_binary_tile(i*2, i*2+1, i*2)`. Integer types use `mul_int_tile` instead.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Needed to understand the SFPU dispatch mechanism (how the SFPU function is called per-face).
   **Key Information**: The params wrapper iterates over 4 faces in RC mode, calling the SFPU function 4 times with `TTI_SETRWC` between faces to advance the DEST pointer.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu.h`
   **Reason**: Needed to understand SFPU start/done/init infrastructure.
   **Key Information**: `_llk_math_eltwise_binary_sfpu_start_` sets DEST write address and stalls for SFPU; `_llk_math_eltwise_binary_sfpu_init_` configures SFPU config reg, address modes, and resets counters.

4. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Needed to understand the `_sfpu_binary_init_` function referenced by the metal-layer `ckernel_sfpu_binary.h`.
   **Key Information**: For MUL, `_sfpu_binary_init_` is a no-op (only DIV/POW/XLOGY require special initialization).
