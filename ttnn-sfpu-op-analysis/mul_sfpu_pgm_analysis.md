# MUL (Binary SFPU) -- Operation Analysis

## Overview

**Operation**: Element-wise multiply (MUL) via the binary SFPU program factory
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`
**Namespace**: `ttnn::operations::binary`
**Class**: `BinaryDeviceOperation::ElementWiseMultiCoreSfpu`

The MUL operation performs element-wise multiplication of two input tensors using the SFPU (Special Function Processing Unit) rather than the FPU (matrix unit). This SFPU path is selected when both input tensors have matching dimensions and matching data types from the set {FLOAT32, INT32, UINT32, UINT16}. The key reason this operation exists on the SFPU -- rather than using the FPU's native `mul_tiles` -- is to support 32-bit precision natively. The FPU operates on bfloat16 internally, so for FLOAT32 or integer types, the SFPU provides a higher-fidelity element-wise multiply by operating directly in the destination register accumulator space.

### When is ElementWiseMultiCoreSfpu selected?

The factory selection logic in `binary_device_operation.cpp` calls `utils::is_binary_sfpu_op(op, dtype1, dtype2)`. For `BinaryOpType::MUL`, this returns `true` when:
- `a == b` (both dtypes match), AND
- `a` is one of: `FLOAT32`, `INT32`, `UINT32`, `UINT16`

Additionally, both tensors must have the same height and width (no broadcasting).

---

## Program Structure

### Kernel Registration

The program factory registers three kernels:

| Kernel Role | File Path | Config Type |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | `ReaderDataMovementConfig` |
| Compute | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | `ComputeConfig` |
| Writer | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (default) or `writer_unary_sharded_blocks_interleaved_start_id.cpp` (block/width sharded, non-sharded output) | `WriterDataMovementConfig` |

### Circular Buffer Configuration

| CB Index | Name | Purpose | Tile Count |
|---|---|---|---|
| `c_0` | `cb_src0` | Input tensor A | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` |
| `c_1` | `cb_src1` | Input tensor B | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` |
| `c_2` | `cb_out0` | Output tensor | Sharded/block-width: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` |
| `c_3` | `cb_interm` | Interim for pre-scaled input A | Only created if `SFPU_OP_INIT_PRE_IN0_0` is defined; `max_block_size` tiles |
| `c_4` | `cb_interm2` | Interim for pre-scaled input B | Only created if `SFPU_OP_INIT_PRE_IN1_0` is defined; `max_block_size` tiles |

For a plain MUL operation (no fused activations), neither `SFPU_OP_INIT_PRE_IN0_0` nor `SFPU_OP_INIT_PRE_IN1_0` are defined, so interim CBs c_3 and c_4 are NOT created. The compute kernel aliases `cb_inp0 = cb_in0` and `cb_inp1 = cb_in1` in this case.

### Compile-Time Defines for MUL

The `get_defines_fp32()` function in `binary_op_utils.cpp` generates the following defines for floating-point MUL (when dtypes are not integer):

```cpp
BINOP_INIT       = "mul_binary_tile_init();"
BINARY_SFPU_OP   = "mul_binary_tile(i*2, i*2+1, i*2);"
```

For integer MUL (e.g., INT32 x INT32):

```cpp
MUL_INT_INIT     = "mul_int_tile_init<DataFormat::Int32>();"
BINARY_SFPU_OP   = "mul_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2);"
```

### UnpackToDestMode

For MUL (not POWER), all input CBs use `UnpackToDestMode::UnpackToDestFp32`. This means input data is unpacked directly into the full-precision DEST registers, bypassing the normal SRCA/SRCB registers. This is critical because the SFPU operates on data already in DEST.

### fp32_dest_acc_en

Enabled when the output dtype is `Float32`, `Int32`, or `UInt32`. This controls whether the SFPU kernel performs software bfloat16 RNE rounding or operates purely in float32.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`

#### Annotated Reader Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args set by set_eltwise_binary_runtime_args<>
    uint32_t src0_addr = get_arg_val<uint32_t>(0);       // DRAM address of input tensor A
    uint32_t src1_addr = get_arg_val<uint32_t>(1);       // DRAM address of input tensor B
    uint32_t num_tiles = get_arg_val<uint32_t>(2);        // total tiles this core must read
    uint32_t start_id = get_arg_val<uint32_t>(3);         // starting tile ID for this core
    uint32_t block_height = get_arg_val<uint32_t>(4);     // shard block height in tiles (used for block/width sharded)
    uint32_t block_width = get_arg_val<uint32_t>(5);      // shard block width in tiles
    uint32_t num_cores_y = get_arg_val<uint32_t>(6);      // num shards per width (for stride calculation)

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;     // CB for input A
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;     // CB for input B
    constexpr bool block_or_width_sharded = get_compile_time_arg_val(0) == 1;  // compile-time sharding mode flag

    // TensorAccessor compile-time args are conditionally instantiated based on sharding defines
#if !defined(IN0_SHARDED) && !defined(IN1_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
#elif !defined(IN0_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
#elif !defined(IN1_SHARDED)
    constexpr auto src1_args = TensorAccessorArgs<1>();
#endif

    // For sharded inputs: data is already in L1, just make the CB visible to compute kernel
#ifdef IN0_SHARDED
    cb_reserve_back(cb_id_in0, num_tiles);   // reserve all tiles at once (already in L1)
    cb_push_back(cb_id_in0, num_tiles);       // publish them immediately
#else
    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const auto s0 = TensorAccessor(src0_args, src0_addr, src0_tile_bytes);
#endif
#ifdef IN1_SHARDED
    cb_reserve_back(cb_id_in1, num_tiles);
    cb_push_back(cb_id_in1, num_tiles);
#else
    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto s1 = TensorAccessor(src1_args, src1_addr, src1_tile_bytes);
#endif

#if !(defined IN0_SHARDED && defined IN1_SHARDED)
    constexpr uint32_t onetile = 1;

    if constexpr (block_or_width_sharded) {
        // Block/width sharded layout: tiles are arranged in 2D blocks across cores
        // Must iterate block_height x block_width and stride by num_cores_y * block_width
        uint32_t row_start_tile_id = start_id;
        for (uint32_t h = 0; h < block_height; h++) {
            uint32_t tile_id = row_start_tile_id;
            for (uint32_t w = 0; w < block_width; w++) {
#ifndef IN0_SHARDED
                cb_reserve_back(cb_id_in0, onetile);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(tile_id, s0, l1_write_addr_in0);  // DMA read from DRAM
#endif
#ifndef IN1_SHARDED
                cb_reserve_back(cb_id_in1, onetile);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(tile_id, s1, l1_write_addr_in1);
#endif
                tile_id++;
                noc_async_read_barrier();  // wait for DMA to complete
#ifndef IN0_SHARDED
                cb_push_back(cb_id_in0, onetile);  // make tile available to compute
#endif
#ifndef IN1_SHARDED
                cb_push_back(cb_id_in1, onetile);
#endif
            }
            row_start_tile_id += num_cores_y * block_width;  // stride to next row of blocks
        }
    } else {
        // Interleaved / height-sharded: simple linear tile iteration
        for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
#ifndef IN0_SHARDED
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(tile_id, s0, l1_write_addr_in0);
#endif
#ifndef IN1_SHARDED
            cb_reserve_back(cb_id_in1, onetile);
            l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(tile_id, s1, l1_write_addr_in1);
#endif
            noc_async_read_barrier();
#ifndef IN0_SHARDED
            cb_push_back(cb_id_in0, onetile);
#endif
#ifndef IN1_SHARDED
            cb_push_back(cb_id_in1, onetile);
#endif
        }
    }
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

This is a generic binary SFPU compute kernel that handles many operations through compile-time defines. For MUL, the key defines are `BINOP_INIT` and `BINARY_SFPU_OP`.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"         // FPU binary ops (not used for SFPU MUL)
#include "api/compute/tile_move_copy.h"          // copy_tile, copy_tile_to_dst_init_short
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"     // mul_binary_tile, mul_binary_tile_init
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"            // mul_int_tile for integer MUL path
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/xlogy.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// This macro evaluates to true if either PRE_IN0 or PRE_IN1 prescaling is defined.
// For plain MUL, neither is defined, so PRE_SCALE is false.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime args: number of tile blocks and block size (tiles per block)
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // how many blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // input B circular buffer

    // For MUL without prescaling, cb_inp0 = cb_in0 and cb_inp1 = cb_in1 (no interim CBs)
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;   // interim CB for prescaled input A
#else
    constexpr auto cb_inp0 = cb_in0;              // direct passthrough
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;   // interim CB for prescaled input B
#else
    constexpr auto cb_inp1 = cb_in1;              // direct passthrough
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // output circular buffer

    // Initialize the unpack/pack pipeline for copy_tile operations
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // fused ReLU on output (not used for plain MUL)
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- PRE-SCALING PHASES (skipped for plain MUL) ---
        // SFPU_OP_INIT_PRE_IN0_0: Would copy tiles from cb_in0 to DST, apply a unary SFPU op,
        // pack results to cb_inp0 (c_3). Used by operations like LOGADDEXP that need exp(x) pre-applied.
        // SFPU_OP_INIT_PRE_IN1_0: Same for input B -> cb_inp1 (c_4).

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
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

        // --- MAIN COMPUTATION PHASE ---
        // Wait for both inputs to be available in their respective CBs
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        // Acquire DEST registers for writing
        tile_regs_acquire();
        tile_regs_wait();

        // Copy all input A tiles into DEST at even indices (0, 2, 4, ...)
        // UnpackToDestFp32 mode means copy_tile unpacks directly to DEST (not SRCA/SRCB)
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);  // configure unpack for A's dtype
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);      // CB tile i -> DEST[i*2]
        }

        // Copy all input B tiles into DEST at odd indices (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);  // configure unpack for B's dtype
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);  // CB tile i -> DEST[i*2+1]

            // For MUL, BINOP_INIT expands to: mul_binary_tile_init();
            // This initializes the SFPU address modifiers and config registers for binary MUL
#ifdef BINOP_INIT
            BINOP_INIT
#endif
            // (Other INIT macros for integer ops, bitwise ops, etc. -- not active for float MUL)
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

            // For MUL, BINARY_SFPU_OP expands to: mul_binary_tile(i*2, i*2+1, i*2);
            // This performs: DEST[i*2] = DEST[i*2] * DEST[i*2+1]
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP
#endif

            // Post-operation SFPU chain (e.g., fused activations like typecast)
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Pack the result from DEST[i*2] to the output CB
            pack_tile(i * 2, cb_out0);
        }
        tile_regs_commit();
        tile_regs_release();

        // Release input CBs and publish output
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

#### Annotated Writer Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);   // output DRAM address
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // number of tiles to write
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // starting tile ID

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // output CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    // Sharded output: just wait for compute to finish writing all tiles
    cb_wait_front(cb_id_out, num_pages);
#else
    constexpr uint32_t onepage = 1;
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    // Interleaved output: write one tile at a time via NoC DMA
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onepage);                    // wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_page(i, s, l1_read_addr);            // DMA write to DRAM
        noc_async_writes_flushed();                            // ensure write is dispatched
        cb_pop_front(cb_id_out, onepage);                      // free the CB slot for compute
    }
    noc_async_write_barrier();                                 // wait for all writes to complete
#endif
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for floating-point MUL.

#### SFPU Kernel File

**Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`
**Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`

Both architectures share identical source code for this file.

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE).
// This is a software implementation used when fp32_dest_acc_en is false,
// meaning the output format is bfloat16. The hardware SFPU always computes
// in float32 internally, so we must manually truncate to bf16 precision.
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Reinterpret the float as raw bits for integer manipulation
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract the LSB of the future bf16 mantissa (bit 16 of the float32 representation).
    // This determines the "tie-breaking" direction for RNE: round to even means
    // when the truncated bits are exactly 0x8000, we round toward the nearest even value.
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add the rounding bias: 0x7FFF + lsb.
    // If lower 16 bits > 0x8000: always rounds up (carry propagates).
    // If lower 16 bits < 0x8000: never rounds up (no carry).
    // If lower 16 bits == 0x8000 (exact tie):
    //   lsb=0 -> adds 0x7FFF -> 0xFFFF -> no carry -> stays even (round down)
    //   lsb=1 -> adds 0x8000 -> 0x10000 -> carry -> rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Mask off the lower 16 bits to produce a valid bf16 value stored in the upper 16 bits
    bits = bits & 0xFFFF0000U;

    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

// Generic binary SFPU operation -- used for ADD, SUB, RSUB.
// NOT used for MUL (MUL has its own specialized function below).
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
}

// Specialized MUL implementation with zero-handling for FPU behavior matching.
// This function exists separately from the generic calculate_sfpu_binary because
// MUL requires special handling: in bfloat16 mode, IEEE 754 says 0 * inf = NaN,
// but the Tenstorrent FPU produces 0 * anything = 0. To maintain consistency
// between FPU and SFPU code paths, the SFPU MUL explicitly checks for zero inputs.
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Each tile in DEST occupies 32 rows when accessed via SFPI (64 / SFP_DESTREG_STRIDE = 32).
    // A full tile has 4 faces of 16x16 = 1024 elements. With SFPU_WIDTH of 32 elements per
    // SFPI instruction, we need 32 iterations to process all 1024 elements (32 * 32 = 1024).
    // But this function processes 8 iterations per face (the caller iterates over 4 faces).
    constexpr uint dst_tile_size_sfpi = 32;

    for (int d = 0; d < ITERATIONS; d++) {
        // Load one vector (32 elements) from each input tile in DEST.
        // dst_index_in0 * dst_tile_size_sfpi gives the base row offset for input A's tile.
        // The sfpi::dst_reg[] access uses the current row pointer, which advances via dst_reg++.
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Element-wise multiply: compiles to SFPMUL instruction
        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // When output is bfloat16, we must:
            // 1) Round the float32 result to bfloat16 precision using RNE
            result = float32_to_bf16_rne(result);

            // 2) Force zero output when either input is zero.
            //    This matches FPU behavior where 0 * x = 0 for all x (including inf/NaN).
            //    Without this, SFPU would produce NaN for 0 * inf per IEEE 754.
            //    Uses SFPI conditional execution (v_if/v_endif) which sets condition codes
            //    per lane, so only lanes where the condition is true get the assignment.
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
        }

        // Write the result back to DEST at the output tile location
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the SFPI row pointer to the next set of 32 elements
        sfpi::dst_reg++;
    }
}

// DIV implementation (included for completeness -- not active for MUL)
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1);
        v_if(in1 == 0) {
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                result = std::numeric_limits<float>::infinity();
                result = sfpi::setsgn(result, in0);
            }
            v_endif;
        }
        v_elseif(in0 == in1) { result = sfpi::vConst1; }
        v_endif;
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

// Init function -- delegates to the architecture-specific binary init in tt-llk
template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### LLK Binop Wrapper

**File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` (identical for Blackhole)

```cpp
#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel {

// Init wrapper: sets up SFPU config regs, address modifiers, and resets counters,
// then calls the operation-specific sfpu_binary_init which is a no-op for most binops.
template <bool APPROXIMATE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BINOP>);
}

// Generic binop dispatch (for ADD, SUB, RSUB) -- uses calculate_sfpu_binary
template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0, dst_index1, odst, vector_mode);
}

// MUL-specific dispatch -- uses the specialized calculate_sfpu_binary_mul
// which includes zero-input handling and software bf16 RNE rounding.
// Note: DST_ACCUM_MODE is passed as is_fp32_dest_acc_en template parameter.
template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_mul(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary_mul<APPROXIMATE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0, dst_index1, odst, vector_mode);
}

// DIV-specific dispatch
template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_div(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary_div<APPROXIMATE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0, dst_index1, odst, vector_mode);
}

}  // namespace ckernel
```

#### Public API Entry Point

**File**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`

```cpp
// mul_binary_tile: The function invoked by the BINARY_SFPU_OP macro in the compute kernel.
// DST_ACCUM_MODE is a global compile-time constant derived from fp32_dest_acc_en in ComputeConfig.
ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binop_mul<APPROX, ckernel::BinaryOp::MUL, DST_ACCUM_MODE>(idst0, idst1, odst)));
}

ALWI void mul_binary_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_binop_init<APPROX, ckernel::BinaryOp::MUL>()));
}
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `SFPMUL` (`__builtin_rvtt_sfpmul`) | Core element-wise multiply of two vFloat vectors (32 lanes). Generated by `in0 * in1`. |
| `SFPSETCC` | Set condition codes per lane. Used by `v_if(in0 == 0 \|\| in1 == 0)` for zero-input detection. |
| `SFPENCC` | Enable/disable lanes based on condition codes. Used by `v_if`/`v_endif` conditional execution. |
| `SFPLOAD` | Load vector from DEST register into SFPU LREG. Generated by `dst_reg[idx]` reads. |
| `SFPSTORE` | Store vector from SFPU LREG back to DEST register. Generated by `dst_reg[idx] = result` writes. |
| `SFPAND` | Bitwise AND on integer representation. Used in `float32_to_bf16_rne` for `bits & 0xFFFF0000U`. |
| `SFPIADD` / `SFPADDI` | Integer add. Used in `float32_to_bf16_rne` for `bits + 0x7fffU + lsb`. |
| `SFPSHFT` | Right shift. Used in `float32_to_bf16_rne` for `bits >> 16`. |
| `SFPLZ` | Load zero / immediate. Used for `result = 0.0f` in the zero-input branch. |

#### SFPU Register Usage

- **DEST registers**: The primary data storage. Input A tiles are at even indices (`i*2`), input B tiles at odd indices (`i*2+1`). The result overwrites the input A location (`i*2`). DEST can hold up to 4 tiles per operand in 16-bit mode or 2 tiles per operand in 32-bit mode.
- **LREG (Local Registers)**: SFPU has 8 local vector registers (LREGs). `in0`, `in1`, `result`, and intermediate values in the RNE conversion all occupy LREGs temporarily during computation.
- **dst_reg pointer**: An implicit row pointer into DEST. `dst_reg++` advances it by `SFP_DESTREG_STRIDE` rows (2 rows), and the `_llk_math_eltwise_binary_sfpu_params_` wrapper advances it across faces using `TTI_SETRWC`.
- **Condition code registers**: Used by `v_if`/`v_endif` to mask lanes. Each of the 32 SFPU lanes has an independent enable bit.

#### SFPU Execution Flow

1. **Tile Acquisition**: `tile_regs_acquire()` / `tile_regs_wait()` -- acquires exclusive access to DEST registers.
2. **Unpack Input A to DEST**: `copy_tile(cb_inp0, i, i*2)` unpacks tile `i` from CB c_0 directly into DEST at index `i*2`. The `UnpackToDestFp32` mode means the unpack hardware writes directly to DEST (bypassing SRCA), converting to float32 if needed.
3. **Unpack Input B to DEST**: `copy_tile(cb_inp1, i, i*2+1)` unpacks tile `i` from CB c_1 into DEST at index `i*2+1`.
4. **SFPU Init**: `mul_binary_tile_init()` configures SFPU address modifiers (ADDR_MOD_7 with no auto-increment) and resets math counters.
5. **SFPU Math**: `mul_binary_tile(i*2, i*2+1, i*2)` is called. The `_llk_math_eltwise_binary_sfpu_params_` wrapper:
   a. Calls `_llk_math_eltwise_binary_sfpu_start_` to set the DEST write address and stall until math is ready.
   b. Iterates over 4 faces (in `VectorMode::RC`), calling `calculate_sfpu_binary_mul` for each face with `ITERATIONS=8`.
   c. Each iteration: loads 32 elements from each input tile in DEST, multiplies them (`SFPMUL`), optionally rounds to bf16 and handles zeros, stores result back to DEST.
   d. Calls `_llk_math_eltwise_binary_sfpu_done_` to clear DEST address and wait for idle.
6. **Pack**: `pack_tile(i*2, cb_out0)` packs the result from DEST[i*2] into the output CB c_2.
7. **Release**: `tile_regs_commit()` / `tile_regs_release()` -- releases DEST registers.
8. **CB Management**: `cb_pop_front` frees input CB slots; `cb_push_back` publishes output tiles.

#### SFPU Configuration

- **`fp32_dest_acc_en`**: Controls whether DEST accumulates in full float32. When true, the `is_fp32_dest_acc_en` template parameter is true, and the bf16 RNE rounding and zero-input handling are skipped. When false, software bf16 rounding is applied.
- **`UnpackToDestFp32`**: Set on all input CBs (c_0, c_1, c_3, c_4). Causes the unpacker to write directly to DEST in float32 format, which is required for SFPU operations since the SFPU reads from DEST, not from SRCA/SRCB.
- **`APPROX`**: A global compile-time constant. For MUL, `APPROXIMATION_MODE` is passed through but does not change behavior (MUL is exact; approximation mode is more relevant for operations like reciprocal or exp).
- **`DST_ACCUM_MODE`**: Derived from `fp32_dest_acc_en`. Passed as the `is_fp32_dest_acc_en` template parameter to `calculate_sfpu_binary_mul`.
- **Math fidelity**: Not directly relevant to SFPU MUL (math fidelity settings affect the FPU matrix unit, not SFPU).

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The `ckernel_sfpu_binary.h` source is identical across both architectures. The `SFPMUL` instruction behaves the same way on both.
- **Blackhole `sfpmul24`**: Blackhole introduces a `sfpmul24` instruction for 24-bit precision multiply, but the standard `mul_binary_tile` path does not use it. The standard path uses full-precision `SFPMUL`.
- **DEST register capacity**: In 32-bit accumulation mode, DEST holds fewer tiles (2 per operand vs 4). The `max_block_size` calculation in the program factory accounts for this, but the SFPU kernel itself uses the same indexing scheme regardless.
- **The `_llk_math_eltwise_binary_sfpu_init_` and `_llk_math_eltwise_binary_sfpu_params_` implementations** are identical across Wormhole B0 and Blackhole per the tt-llk submodule, indicating shared SFPU control-flow logic.

---

## Runtime Arguments

### Compute Kernel Runtime Args

| Index | Name | Description |
|---|---|---|
| 0 | `per_core_block_cnt` | Number of tile blocks to process on this core |
| 1 | `per_core_block_size` | Number of tiles per block |

These are computed by `set_eltwise_binary_runtime_args` based on work splitting across cores.

### Reader Kernel Runtime Args

| Index | Name | Description |
|---|---|---|
| 0 | `src0_addr` | DRAM address of input tensor A |
| 1 | `src1_addr` | DRAM address of input tensor B |
| 2 | `num_tiles` | Total tiles for this core |
| 3 | `start_id` | Starting tile ID |
| 4 | `block_height` | Block height in tiles (sharded) |
| 5 | `block_width` | Block width in tiles (sharded) |
| 6 | `num_cores_y` | Number of shards per width |

### Writer Kernel Runtime Args

| Index | Name | Description |
|---|---|---|
| 0 | `dst_addr` | DRAM address of output tensor |
| 1 | `num_pages` | Number of tiles to write |
| 2 | `start_id` | Starting tile ID |

---

## Work Distribution

Work is distributed across cores using `split_work_to_cores()` for interleaved layouts, or by shard specification for sharded layouts. The `set_eltwise_binary_runtime_args` template function handles both cases:

- **Interleaved**: Tiles are divided evenly across available cores. Core group 1 may get one more tile than core group 2 to handle remainders.
- **Height sharded**: Each core processes its own shard. `num_tiles_per_shard` tiles per core.
- **Block/width sharded**: Uses 2D block iteration with `block_height * block_width` tiles per core, striding by `num_shards_per_width * block_width` between rows.

---

## Call Chain Summary (for floating-point MUL)

```
BINARY_SFPU_OP macro
  = mul_binary_tile(i*2, i*2+1, i*2)                       [eltwise_binary_sfpu.h]
    -> llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(...)
                                                             [llk_math_eltwise_binary_sfpu_binop.h]
       -> _llk_math_eltwise_binary_sfpu_params_<APPROX>(
              calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8, DST_ACCUM_MODE>, ...)
                                                             [tt-llk: llk_math_eltwise_binary_sfpu_params.h]
          -> _llk_math_eltwise_binary_sfpu_start_(0)        // set DEST write addr, stall
          -> for each face (4x in VectorMode::RC):
               calculate_sfpu_binary_mul(dst_in0, dst_in1, dst_out)
                                                             [ckernel_sfpu_binary.h]
                 -> for d in 0..7:
                      in0 = dst_reg[in0_offset]             // SFPLOAD
                      in1 = dst_reg[in1_offset]             // SFPLOAD
                      result = in0 * in1                     // SFPMUL
                      if (!fp32_dest_acc_en):
                        result = float32_to_bf16_rne(result) // SFPSHFT, SFPAND, SFPIADD
                        v_if(in0==0 || in1==0): result=0    // SFPSETCC, SFPENCC, SFPLZ
                      dst_reg[out_offset] = result           // SFPSTORE
                      dst_reg++                              // advance row pointer
               TTI_SETRWC(...)                               // advance to next face
          -> _llk_math_eltwise_binary_sfpu_done_()          // clear DEST addr, wait idle
```

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Binary SFPU program factory architecture, kernel registration patterns, CB configuration
- **tenstorrent/tt-llk**: `_llk_math_eltwise_binary_sfpu_params_` face iteration logic, `_llk_math_eltwise_binary_sfpu_init_` setup (address modifiers, counter reset)
- **tenstorrent/sfpi**: `SFPMUL` instruction mapping via `__builtin_rvtt_sfpmul`, `vFloat` operator overloads, SFPU emulator reference

### Confluence References

Not consulted for this analysis. The DeepWiki sources provided sufficient detail for the SFPU instructions used by MUL.

### Glean References

Not consulted for this analysis. The codebase source files and DeepWiki provided complete coverage of the MUL SFPU kernel implementation.
