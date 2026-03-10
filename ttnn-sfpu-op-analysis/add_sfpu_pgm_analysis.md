# SFPU Operation Analysis: ADD (Legacy Element-Wise Binary SFPU)

## Overview

This document provides a comprehensive structural analysis of the binary ADD operation as implemented through the SFPU (Special Function Processing Unit) path in the legacy element-wise multi-core program factory. The SFPU path is used instead of the FPU path when operating in FP32 destination accumulation mode or when mixed-precision operands require explicit data handling that the SFPU's scalar vector pipeline supports more flexibly.

**Operation**: `BinaryOpType::ADD`
**Program Factory**: `ElementWiseMultiCoreSfpu`
**Program Factory File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

The SFPU ADD for floating-point types resolves to `add_binary_tile(i*2, i*2+1, i*2)`, which in turn calls `_calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>(...)`. The core math is a single SFPI instruction: `result = in0 + in1`, which maps to the hardware `sfpadd` instruction. For integer types (INT32, UINT32, UINT16), a separate `add_int_tile` path is used instead.

---

## Program Factory Structure

### Factory Class
- **Namespace**: `ttnn::operations::binary`
- **Class**: `BinaryDeviceOperation::ElementWiseMultiCoreSfpu`
- **Methods**:
  - `create(...)` -- Builds the full program (CBs, kernels, runtime args)
  - `override_runtime_arguments(...)` -- Updates runtime args for cached program reuse

### Operation Attributes Used
| Attribute | Purpose |
|---|---|
| `binary_op_type` | Selects the SFPU binary operation (ADD, SUB, MUL, etc.) |
| `activations` | Optional fused post-op activations (e.g., RELU) |
| `input_tensor_a_activation` | Optional pre-processing activation on input A |
| `worker_grid` | CoreRangeSet of cores to dispatch to |

---

## Circular Buffer Configuration

| CB Index | Name | Role | Size (non-sharded) | Size (sharded) |
|---|---|---|---|---|
| `c_0` | `cb_src0` | Input A | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_1` | `cb_src1` | Input B | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_2` | `cb_output` | Output | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_3` | `cb_inp0` (interim) | Pre-processed input A | `max_block_size * tile_size` | Same |
| `c_4` | `cb_inp1` (interim) | Pre-processed input B | `max_block_size * tile_size` | Same |

**Notes on CB c_3 and c_4**: These interim CBs are only created when the compile-time defines `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` are present. For plain ADD, neither is defined, so only c_0, c_1, and c_2 are used. Operations like LOGADDEXP or DIV use these interim buffers for pre-scaling (e.g., applying EXP or RECIP to inputs before the binary operation).

**Sharding behavior**: When an input or output tensor is sharded, its CB is globally allocated to the tensor's buffer address. This avoids data movement -- the reader/writer simply marks the CB as ready without actually copying data.

### Data Format Handling

The program factory determines data formats from tensor dtypes:
- `src0_cb_data_format` from input A dtype
- `src1_cb_data_format` from input B dtype
- `dst_cb_data_format` from output dtype

FP32 destination accumulation is enabled when the output format is Float32, Int32, or UInt32. For non-POWER operations, all CBs use `UnpackToDestMode::UnpackToDestFp32`, meaning the unpack hardware converts data to FP32 before placing it in destination registers.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
**Type**: ReaderDataMovementConfig

#### Annotated Reader Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments: buffer addresses, tile count, starting tile ID, and block dimensions
    uint32_t src0_addr = get_arg_val<uint32_t>(0);     // DRAM address of input tensor A
    uint32_t src1_addr = get_arg_val<uint32_t>(1);     // DRAM address of input tensor B
    uint32_t num_tiles = get_arg_val<uint32_t>(2);      // Total tiles this core processes
    uint32_t start_id = get_arg_val<uint32_t>(3);       // Starting tile index for this core
    uint32_t block_height = get_arg_val<uint32_t>(4);   // Used for block/width sharded layouts
    uint32_t block_width = get_arg_val<uint32_t>(5);    // Used for block/width sharded layouts
    uint32_t num_cores_y = get_arg_val<uint32_t>(6);    // Number of shards per width dimension

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;   // CB for input A
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;   // CB for input B
    // Compile-time arg: whether layout is block or width sharded
    constexpr bool block_or_width_sharded = get_compile_time_arg_val(0) == 1;

    // TensorAccessor compile-time args are conditionally compiled based on sharding defines.
    // When an input is sharded, its TensorAccessor is not needed because data is already in L1.
#if !defined(IN0_SHARDED) && !defined(IN1_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
#elif !defined(IN0_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
#elif !defined(IN1_SHARDED)
    constexpr auto src1_args = TensorAccessorArgs<1>();
#endif

    // For sharded inputs, just mark the CB as having all tiles ready.
    // The CB is globally allocated to the tensor's buffer, so no DMA needed.
#ifdef IN0_SHARDED
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in0, num_tiles);
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
        // Block/width sharded: iterate through a 2D block of tiles.
        // row_start_tile_id jumps by num_cores_y * block_width between rows
        // because each row of shards is distributed across cores.
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
                noc_async_read_barrier();  // Wait for DMA to complete before publishing
#ifndef IN0_SHARDED
                cb_push_back(cb_id_in0, onetile);
#endif
#ifndef IN1_SHARDED
                cb_push_back(cb_id_in1, onetile);
#endif
            }
            row_start_tile_id += num_cores_y * block_width;
        }
    } else {
        // Height sharded or interleaved: simple linear tile iteration
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

### Writer Kernel

**File** (default path): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
**Alternative** (block/width sharded, non-sharded output): `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
**Type**: WriterDataMovementConfig

#### Annotated Writer Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);   // DRAM address of output tensor
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of tiles to write
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting tile index

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    // Sharded output: just wait for compute to fill all tiles.
    // The CB is globally allocated to the output buffer, so data is already in place.
    cb_wait_front(cb_id_out, num_pages);
#else
    constexpr uint32_t onepage = 1;
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onepage);              // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_page(i, s, l1_read_addr);       // DMA write tile to DRAM
        noc_async_writes_flushed();                      // Ensure write is dispatched
        cb_pop_front(cb_id_out, onepage);                // Free the CB slot
    }
    noc_async_write_barrier();  // Final barrier to ensure all writes complete
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`
**Type**: ComputeConfig with SFPU defines

This is the central compute kernel that orchestrates the SFPU binary operation. It is parameterized entirely through compile-time `#define` macros set by the host-side `get_defines_fp32()` function.

#### Compile-Time Defines for ADD (Floating-Point)

For `BinaryOpType::ADD` with floating-point inputs, `get_defines_fp32()` produces:
- `BINOP_INIT` = `add_binary_tile_init();`
- `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2);`

No `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are generated, so the pre-scaling code paths are inactive.

#### Compile-Time Defines for ADD (Integer Types)

For integer inputs (e.g., both INT32), `get_defines_fp32()` produces:
- `ADD_INT_INIT` = `add_int_tile_init();`
- `BINARY_SFPU_OP` = `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2);`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"     // Provides add_binary_tile, sub_binary_tile, etc.
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"             // Provides add_int_tile for integer addition
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

// PRE_SCALE is true if either input requires pre-processing (e.g., EXP for LOGADDEXP)
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime arguments from the host: how many blocks and how many tiles per block
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // Number of tile blocks
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);  // Tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // Raw input A from reader
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // Raw input B from reader

    // If pre-scaling is needed, use interim CBs; otherwise alias directly to input CBs
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;   // Pre-processed input A
#else
    constexpr auto cb_inp0 = cb_in0;              // For ADD: no pre-processing, use raw input directly
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;   // Pre-processed input B
#else
    constexpr auto cb_inp1 = cb_in1;              // For ADD: no pre-processing, use raw input directly
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // Output CB

    // Initialize common unary op state (sets up packer/unpacker for the CB formats)
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    // For ADD with fused RELU activation (single RELU), use hardware pack-time RELU
    // instead of an SFPU post-op. This is faster because it happens during pack.
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- PRE-SCALING PHASE (inactive for plain ADD) ---
        // When SFPU_OP_INIT_PRE_IN0_0 is defined, this block copies input A tiles
        // from cb_in0 to DST, applies an SFPU function (e.g., EXP), and packs
        // results to cb_inp0 (c_3). Similarly for input B with PRE_IN1_0.
#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        cb_wait_front(cb_in0, per_core_block_size);
        cb_reserve_back(cb_inp0, per_core_block_size);
        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN0_0          // e.g., exp_tile_init() for LOGADDEXP
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, i);     // Copy tile from CB to DST register i
            SFPU_OP_FUNC_PRE_IN0_0       // e.g., exp_tile(i) -- apply SFPU to DST[i]
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp0);       // Pack DST[i] into interim CB
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

        // --- MAIN BINARY OPERATION PHASE ---
        // Wait for both inputs and reserve output space
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();   // Acquire exclusive access to DST registers
        tile_regs_wait();      // Wait for DST registers to be available

        // Copy input A tiles into even DST slots (0, 2, 4, ...)
        // The _with_dt variant reconfigures the unpacker for the source CB's data type
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);      // CB_inp0[i] -> DST[i*2]
        }

        // Copy input B tiles into odd DST slots (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);  // CB_inp1[i] -> DST[i*2+1]

            // Initialize the SFPU operation (one-time per tile for some ops)
            // For ADD: expands to add_binary_tile_init();
#ifdef BINOP_INIT
            BINOP_INIT
#endif
#ifdef ADD_INT_INIT
            ADD_INT_INIT       // For integer ADD: expands to add_int_tile_init();
#endif
            // (Other INIT macros for SUB_INT, MUL_INT, etc. are inactive for ADD)

            // Execute the binary SFPU operation
            // For floating-point ADD: expands to add_binary_tile(i*2, i*2+1, i*2);
            // This reads DST[i*2] (in0) and DST[i*2+1] (in1), computes in0+in1,
            // and writes the result back to DST[i*2]
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP
#endif

            // (SFPU_OP_INIT_0/FUNC_0 and SFPU_OP_CHAIN_0 are for fused post-ops)
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0     // e.g., for BIAS_GELU: gelu_tile_init()
            SFPU_OP_FUNC_0     // e.g., gelu_tile(i*2)
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0    // Chained post-ops (e.g., typecast)
#endif

            // Pack the result from DST[i*2] into the output CB
            pack_tile(i * 2, cb_out0);
        }

        tile_regs_commit();    // Signal that DST writes are complete
        tile_regs_release();   // Release DST register ownership

        // Release input CBs and publish output
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the floating-point ADD case.

#### SFPU Kernel File

The core SFPU binary implementation lives in the tt-llk submodule:
- **Wormhole**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`
- **Blackhole**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`

The metal-side wrappers (which call into tt-llk) are at:
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`

#### Annotated SFPU Kernel Source (tt-llk, Wormhole B0)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"                    // SFPI C++ wrapper around SFPU hardware intrinsics

namespace ckernel
{
namespace sfpu
{

// The main templated function for all simple binary SFPU operations.
// Templated on the BinaryOp enum to select the operation at compile time.
// ITERATIONS=8 processes 8 rows per call (one 16x16 face).
// The _llk_math_eltwise_binary_sfpu_params_ wrapper calls this 4 times
// in RC mode to cover all 4 faces of a 32x32 tile.
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();

    // SFPU microcode: process ITERATIONS rows of the current face.
    // Each iteration processes one row of the SFPU vector width (32 elements
    // processed as a vector lane of width determined by the hardware).
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile in Dest occupies 32 rows when accessed via SFPI
        // (64 rows / SFP_DESTREG_STRIDE of 2)
        constexpr std::uint32_t dst_tile_size_sfpi = 32;

        // Load one vector-width row from each input tile in the DST register file.
        // dst_reg[index] accesses the current row at offset (index * dst_tile_size_sfpi)
        // from the base address that auto-increments via dst_reg++.
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        // Compile-time dispatch based on BinaryOp enum.
        // For ADD, this is the only active branch.
        if constexpr (BINOP == BinaryOp::ADD)
        {
            // Element-wise floating-point addition.
            // Maps to the SFPU sfpadd instruction via the SFPI operator+ overload.
            // sfpadd performs: for each lane i: result[i] = in0[i] + in1[i]
            result = in0 + in1;
        }
        else if constexpr (BINOP == BinaryOp::SUB)
        {
            result = in0 - in1;
        }
        else if constexpr (BINOP == BinaryOp::MUL)
        {
            result = in0 * in1;
        }
        else if constexpr (BINOP == BinaryOp::DIV)
        {
            result = in0 * _sfpu_reciprocal_<2>(in1);
        }
        else if constexpr (BINOP == BinaryOp::RSUB)
        {
            result = in1 - in0;
        }
        else if constexpr (BINOP == BinaryOp::POW)
        {
            result = _calculate_sfpu_binary_power_(in0, in1);
        }
        else if constexpr (BINOP == BinaryOp::XLOGY)
        {
            v_if ((in1 < 0.0f) || (in1 == nan))
            {
                result = nan;
            }
            v_else
            {
                sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in1;
                _calculate_log_body_<false>(0, dst_index_out);
                result = sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] * in0;
            }
            v_endif;
        }

        // Write the result back to the output tile position in DST
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance to the next row within the face.
        // dst_reg++ increments the base row pointer by SFP_DESTREG_STRIDE.
        sfpi::dst_reg++;
    }
}

// Initialization function for binary SFPU operations.
// For ADD, no special initialization is needed (no reciprocal tables, no log tables).
template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();   // Load reciprocal LUT for Newton-Raphson
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>(); // Load log coefficients
    }
    // For ADD, SUB, MUL, RSUB: no initialization required
}

} // namespace sfpu
} // namespace ckernel
```

#### Metal-Side Wrapper (ckernel_sfpu_binary.h in tt_metal)

The metal-side `ckernel_sfpu_binary.h` provides a thin forwarding layer:

```cpp
// calculate_sfpu_binary just forwards to the tt-llk implementation
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
}
```

Note: The `is_fp32_dest_acc_en` template parameter is present but unused for the generic binary path. The specialized `calculate_sfpu_binary_mul` and `calculate_sfpu_binary_div` variants in the metal wrapper do use it to conditionally apply BF16 rounding.

#### SFPU Instructions Used

| Instruction/Intrinsic | SFPI Mapping | Description |
|---|---|---|
| `sfpadd` | `vFloat operator+(vFloat, vFloat)` | Element-wise floating-point addition across all SFPU vector lanes |
| `SFPLOADI` (implicit) | `sfpi::dst_reg[offset]` read | Load a vector row from DST register file into an SFPU local register |
| `SFPSTORE` (implicit) | `sfpi::dst_reg[offset] = val` write | Store a vector from SFPU local register back to DST register file |

For the ADD operation specifically, the instruction sequence per row is minimal:
1. Load `in0` from DST (SFPLOADI-equivalent)
2. Load `in1` from DST (SFPLOADI-equivalent)
3. `sfpadd` to compute `in0 + in1`
4. Store result to DST (SFPSTORE-equivalent)

#### SFPU Register Usage

| Register | Usage |
|---|---|
| DST[i*2 * 32 + row] | Input tile A, accessed via `dst_reg[dst_index_in0 * 32]` |
| DST[(i*2+1) * 32 + row] | Input tile B, accessed via `dst_reg[dst_index_in1 * 32]` |
| DST[i*2 * 32 + row] (output) | Result overwrites input A position, `dst_reg[dst_index_out * 32]` |
| SFPU local registers (L0-L7) | Temporary storage for `in0`, `in1`, `result` during SFPI execution |

The DST register file is shared between the FPU and SFPU. Each 32x32 tile occupies 32 SFPI-addressable rows (because SFP_DESTREG_STRIDE = 2, so 64 physical rows / 2 = 32 logical rows). Two tiles are loaded simultaneously (in0 at even slot, in1 at odd slot), and the result overwrites the even slot.

#### SFPU Execution Flow

1. **Reader fills input CBs**: The reader kernel DMA-reads tiles from DRAM into `cb_in0` (c_0) and `cb_in1` (c_1), one tile at a time, publishing each via `cb_push_back`.

2. **Compute waits for inputs**: `cb_wait_front(cb_inp0, per_core_block_size)` and `cb_wait_front(cb_inp1, per_core_block_size)` block until the reader has produced enough tiles.

3. **Tile register acquisition**: `tile_regs_acquire()` + `tile_regs_wait()` acquire exclusive access to the DST register file.

4. **Unpack input A to DST (even slots)**: `copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0)` reconfigures the unpacker for cb_inp0's data format, then `copy_tile(cb_inp0, i, i*2)` unpacks tile `i` from the CB into DST slot `i*2`. This uses the A2D (Any-to-Dest) hardware path.

5. **Unpack input B to DST (odd slots)**: Similarly, `copy_tile(cb_inp1, i, i*2+1)` unpacks into DST slot `i*2+1`.

6. **SFPU initialization**: `add_binary_tile_init()` calls through to `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()`, which:
   - Calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` to configure SFPU config registers, set up address modifiers (ADDR_MOD_7 with zero increments), and reset counters.
   - Calls `sfpu_binary_init<APPROX, BinaryOp::ADD>()` which is a no-op for ADD (no LUT initialization needed).

7. **SFPU binary execution**: `add_binary_tile(i*2, i*2+1, i*2)` calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(i*2, i*2+1, i*2)`, which calls `_llk_math_eltwise_binary_sfpu_params_`. This function:
   - Calls `_llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0)` to set the DST write address and stall until the math pipeline is ready.
   - In `VectorMode::RC` (default), loops over all 4 faces of the 32x32 tile. For each face, calls `calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8>()` which processes 8 rows.
   - Between faces, issues `TTI_SETRWC` to advance the DST row counter by 16 rows (two increments of 8).
   - Calls `_llk_math_eltwise_binary_sfpu_done_()` to clear the DST address and wait for SFPU completion.

8. **Pack result to output CB**: `pack_tile(i*2, cb_out0)` packs the result from DST slot `i*2` into the output circular buffer.

9. **Release and publish**: `tile_regs_commit()` + `tile_regs_release()` release DST ownership. `cb_pop_front` frees input CB slots, `cb_push_back` publishes output tiles.

10. **Writer drains output CB**: The writer kernel waits on `cb_wait_front(cb_out0, 1)`, reads the tile, and DMA-writes it to DRAM via `noc_async_write_page`.

#### SFPU Configuration

| Configuration | Value | Reason |
|---|---|---|
| `fp32_dest_acc_en` | `true` when output is Float32/Int32/UInt32 | Enables 32-bit accumulation in DST registers |
| `UnpackToDestMode` | `UnpackToDestFp32` for all CBs (non-POWER ops) | Forces unpack to FP32 in DST for SFPU precision |
| `APPROX` | Compile-time bool from ComputeConfig | Controls approximation mode (unused for ADD) |
| `VectorMode` | `RC` (Row-Column, all 4 faces) | Processes entire 32x32 tile |
| `ADDR_MOD_7` | srca=0, srcb=0, dest=0 | No auto-increment on address modifiers |
| ITERATIONS | 8 | Processes 8 rows per face (8 * 4 faces = 32 rows total) |

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `_calculate_sfpu_binary_` are **identical** for the ADD operation. Both use the same `sfpi::vFloat` operator+ which maps to `sfpadd`. The only difference between architectures is in the SFPI backend:

- **Wormhole B0**: `sfpadd` maps to `__builtin_rvtt_wh_sfpadd`
- **Blackhole**: `sfpadd` maps to `__builtin_rvtt_bh_sfpadd`

Both implement IEEE 754 floating-point addition across all vector lanes. The functional behavior is identical.

The metal-side `ckernel_sfpu_binary.h` wrappers in both `wormhole_b0` and `blackhole` directories also include `calculate_sfpu_binary_mul` and `calculate_sfpu_binary_div` which have additional logic (BF16 rounding via `float32_to_bf16_rne`, special-case handling for zero/infinity), but these are not invoked for the ADD path.

---

## Macro-Driven Define System

The program factory uses `get_defines_fp32()` from `binary_op_utils.cpp` to generate compile-time defines that parameterize the compute kernel. This is a key architectural pattern: a single generic compute kernel source file serves all binary SFPU operations through preprocessor defines.

### Define Generation for ADD

```
Input: BinaryOpType::ADD, input_a_dtype, input_b_dtype
```

**Floating-point path** (e.g., BFLOAT16 x BFLOAT16):
```
BINOP_INIT     -> "add_binary_tile_init();"
BINARY_SFPU_OP -> "add_binary_tile(i*2, i*2+1, i*2);"
```

**Integer path** (e.g., INT32 x INT32):
```
ADD_INT_INIT   -> "add_int_tile_init();"
BINARY_SFPU_OP -> "add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2);"
```

**With fused RELU** (single RELU activation on ADD):
```
PACK_RELU -> "1"
```
This uses hardware pack-time RELU rather than an SFPU post-op, which is an optimization specific to ADD+RELU.

### DST Register Layout

The compute kernel uses an interleaved layout in DST:
```
DST[0] = input A tile 0    (accessed as i*2 where i=0)
DST[1] = input B tile 0    (accessed as i*2+1 where i=0)
DST[2] = input A tile 1    (accessed as i*2 where i=1)
DST[3] = input B tile 1    (accessed as i*2+1 where i=1)
...
```
The SFPU operation reads from `DST[i*2]` and `DST[i*2+1]` and writes the result to `DST[i*2]`, overwriting input A. The packer then reads from `DST[i*2]`.

---

## Runtime Arguments

### Reader Kernel Runtime Args
| Index | Name | Description |
|---|---|---|
| 0 | `src0_addr` | DRAM address of input tensor A |
| 1 | `src1_addr` | DRAM address of input tensor B |
| 2 | `num_tiles` | Number of tiles for this core |
| 3 | `start_id` | Starting tile index |
| 4 | `block_height` | Shard block height (tiles) |
| 5 | `block_width` | Shard block width (tiles) |
| 6 | `num_cores_y` | Number of shards per width |

### Compute Kernel Runtime Args
| Index | Name | Description |
|---|---|---|
| 0 | `per_core_block_cnt` | Number of tile blocks to process |
| 1 | `per_core_block_size` | Number of tiles per block |

### Writer Kernel Runtime Args (non-sharded output)
| Index | Name | Description |
|---|---|---|
| 0 | `dst_addr` | DRAM address of output tensor |
| 1 | `num_pages` | Number of tiles to write |
| 2 | `start_id` | Starting tile index |

---

## Work Distribution

The program factory supports three work distribution strategies:

1. **Interleaved (no sharding)**: Tiles are split evenly across cores using `split_work_to_cores`. Each core processes a contiguous range of tile IDs. Two core groups may exist if tiles don't divide evenly.

2. **Height sharded**: Each core owns a contiguous shard of tiles in the height dimension. The reader marks its CB as ready without DMA.

3. **Block/width sharded**: Tiles are distributed in a 2D grid pattern. The reader uses `block_height`/`block_width` to iterate through a 2D tile block, with `num_cores_y` used to compute stride between rows.

### Program Caching

The `override_runtime_arguments` method enables program caching. On subsequent invocations with the same operation attributes but different tensor buffers, only the runtime arguments (buffer addresses, tile counts) are updated without rebuilding the entire program. The cached program stores kernel handles, CB handles, core grid, and tile sizes.

---

## Call Chain Summary

```
Host: get_defines_fp32(BinaryOpType::ADD, ...)
  -> defines["BINOP_INIT"] = "add_binary_tile_init();"
  -> defines["BINARY_SFPU_OP"] = "add_binary_tile(i*2, i*2+1, i*2);"

Compute kernel: eltwise_binary_sfpu_kernel.cpp
  -> BINOP_INIT expands to: add_binary_tile_init()
      -> llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()
          -> _llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()
              -> sfpu::_init_sfpu_config_reg()
              -> eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()
              -> math::reset_counters(SET_ABD_F)
          -> sfpu_binary_init<APPROX, BinaryOp::ADD>()
              -> _sfpu_binary_init_<APPROX, BinaryOp::ADD>()  // no-op for ADD

  -> BINARY_SFPU_OP expands to: add_binary_tile(i*2, i*2+1, i*2)
      -> llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(i*2, i*2+1, i*2)
          -> _llk_math_eltwise_binary_sfpu_params_<APPROX>(
                calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, false>,
                i*2, i*2+1, i*2, VectorMode::RC)
              -> _llk_math_eltwise_binary_sfpu_start_<DST_SYNC>(0)
                  -> math::set_dst_write_addr(0)
                  -> TTI_STALLWAIT(STALL_SFPU, MATH)
              -> for face in 0..3:
                  -> calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8>(i*2, i*2+1, i*2)
                      -> _calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>(...)
                          -> for row in 0..7:
                              in0 = dst_reg[i*2 * 32]     // SFPLOADI
                              in1 = dst_reg[(i*2+1) * 32]  // SFPLOADI
                              result = in0 + in1           // sfpadd
                              dst_reg[i*2 * 32] = result   // SFPSTORE
                              dst_reg++                    // advance row
                  -> TTI_SETRWC (advance DST counter by 16 rows)
              -> _llk_math_eltwise_binary_sfpu_done_()
                  -> math::clear_dst_reg_addr()
                  -> TTI_STALLWAIT(STALL_CFG, WAIT_SFPU)
```

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Binary eltwise SFPU program factory structure, CB configuration, kernel dispatch patterns
- `tenstorrent/tt-llk`: `_calculate_sfpu_binary_` function structure, `_llk_math_eltwise_binary_sfpu_params_` face iteration logic, LLK API call chain
- `tenstorrent/sfpi`: SFPI operator overloads mapping to SFPU instructions (`vFloat operator+` -> `sfpadd`), local register model (L0-L7), `dst_reg` accessor semantics

### Confluence References
Not consulted for this analysis. The ADD operation uses a single `sfpadd` instruction whose behavior is well-documented in DeepWiki and the source code.

### Glean References
Not consulted for this analysis. The ADD operation does not require confidential hardware specification details beyond what is available in the open-source code.
