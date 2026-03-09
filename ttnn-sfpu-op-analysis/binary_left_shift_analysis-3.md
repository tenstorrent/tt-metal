# Binary Left Shift -- SFPU Operation Analysis

## Operation Overview

**Operation Name**: Binary Left Shift (`BinaryOpType::LEFT_SHIFT`)
**Category**: Elementwise Binary (SFPU)
**Program Factory**: `BinaryDeviceOperation::ElementWiseMultiCoreSfpu`
**Namespace**: `ttnn::operations::binary`
**TTNN API Names**: `ttnn::bitwise_left_shift`, `ttnn::logical_left_shift`

### Functional Description

The Binary Left Shift operation performs an elementwise bitwise left shift: `output[i] = input_a[i] << input_b[i]`. Tensor A provides the values to shift and Tensor B provides the per-element shift amounts. The shift amounts must be integers in the range [0, 31]; values outside this range produce 0.

### Supported Data Types

The operation is classified as an SFPU binary op (routed through `ElementWiseMultiCoreSfpu`) when both input tensors share the same integer data type:
- `Int32` and `Int32`
- `UInt32` and `UInt32`

Additionally, `UInt16` support exists at the kernel level but is not exposed through the SFPU program factory selection path (which requires `Int32`/`Int32` or `UInt32`/`UInt32`).

### Program Factory Selection

`BinaryDeviceOperation::select_program_factory` routes to `ElementWiseMultiCoreSfpu` when:
1. The operation is identified as an SFPU op via `utils::is_binary_sfpu_op(op, dtype_a, dtype_b)` -- for `LEFT_SHIFT` this requires both inputs to be `(Int32, Int32)` or `(UInt32, UInt32)`.
2. The input tensors have the same height and width (no broadcasting needed).
3. No scalar operand is provided.

If broadcasting is required, separate broadcast program factories (`BroadcastHeightAndWidthMultiCore`, etc.) are selected instead.

---

## Program Factory Analysis

**Source File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

### Circular Buffers

| CB Index | Symbolic Name | Role | Data Format | Tile Count |
|----------|---------------|------|-------------|------------|
| `c_0` | `cb_src0` | Input tensor A | Matches `a.dtype()` | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` |
| `c_1` | `cb_src1` | Input tensor B | Matches `b.dtype()` | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` |
| `c_2` | `cb_output` | Output tensor | Matches `output.dtype()` | Sharded/block: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` |
| `c_3` | `cb_interm` | Pre-processing interim for input A | Same as src0 format | `max_block_size` (only allocated if `SFPU_OP_INIT_PRE_IN0_0` defined) |
| `c_4` | `cb_interm2` | Pre-processing interim for input B | Same as src1 format | `max_block_size` (only allocated if `SFPU_OP_INIT_PRE_IN1_0` defined) |

For Binary Left Shift, neither `SFPU_OP_INIT_PRE_IN0_0` nor `SFPU_OP_INIT_PRE_IN1_0` is defined, so `c_3` and `c_4` are not allocated. The compute kernel reads directly from `c_0` and `c_1`.

### Sharding Support

The factory supports three memory configurations:
- **Interleaved**: Both inputs and output in DRAM with interleaved layout. Reader fetches tiles one-at-a-time from DRAM via NoC.
- **Sharded inputs**: Input CBs are backed by the tensor's globally-allocated L1 buffer (zero-copy). The reader simply marks tiles as available.
- **Sharded output**: Output CB is backed by the output tensor's L1 buffer. Writer is a no-op after CB push.
- **Block/width sharding**: Detected via `block_or_width_sharded` flag, affects reader tile traversal order and writer kernel selection.

### Compile-Time Defines for Left Shift

The `utils::get_defines_fp32` function generates these defines for `BinaryOpType::LEFT_SHIFT`:

| Define | Value | Purpose |
|--------|-------|---------|
| `SHIFT_INIT` | `binary_shift_tile_init();` | Initializes the SFPU for shift operations |
| `BINARY_SFPU_OP` | `binary_left_shift_tile<DataFormat::Int32>(i * 2, i * 2 + 1, i * 2);` (for Int32) | Invokes the left shift tile operation |

The `DataFormat` template argument is selected based on input types:
- `Int32` x `Int32` -> `DataFormat::Int32`
- `UInt32` x `UInt32` -> `DataFormat::UInt32`
- `UInt16` x `UInt16` -> `DataFormat::UInt16`

### Unpack-to-Dest Mode

For all non-POWER SFPU binary ops, the factory sets `UnpackToDestMode::UnpackToDestFp32` on all input CBs (`c_0`, `c_1`, `c_3`, `c_4`). This is critical because SFPU operations operate on data in the Destination register, and unpack-to-dest mode bypasses the FPU pipeline by loading unpacked tiles directly into Dest.

### FP32 Dest Accumulation

`fp32_dest_acc_en` is set to `true` when the output format is `Float32`, `Int32`, or `UInt32`. For integer shift operations this is always true, ensuring the full 32-bit Dest register width is used.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`

#### Annotated Reader Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// This reader kernel handles both interleaved and sharded inputs for binary operations.
// For sharded inputs, it simply marks tiles as available in the CB (zero-copy from L1).
// For interleaved inputs, it reads tiles from DRAM via NoC one at a time.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments passed from the program factory's set_eltwise_binary_runtime_args
    uint32_t src0_addr = get_arg_val<uint32_t>(0);  // DRAM address of input tensor A
    uint32_t src1_addr = get_arg_val<uint32_t>(1);  // DRAM address of input tensor B
    uint32_t num_tiles = get_arg_val<uint32_t>(2);   // Total tiles this core processes
    uint32_t start_id = get_arg_val<uint32_t>(3);    // Starting tile ID for this core
    uint32_t block_height = get_arg_val<uint32_t>(4); // Block height for block/width sharding
    uint32_t block_width = get_arg_val<uint32_t>(5);  // Block width for block/width sharding
    uint32_t num_cores_y = get_arg_val<uint32_t>(6);  // Number of cores in Y dim (for stride calc)

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr bool block_or_width_sharded = get_compile_time_arg_val(0) == 1;

    // TensorAccessor compile-time args are conditionally included based on sharding defines.
    // Only non-sharded inputs need accessor args (sharded inputs use globally-allocated CB).
#if !defined(IN0_SHARDED) && !defined(IN1_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
#elif !defined(IN0_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
#elif !defined(IN1_SHARDED)
    constexpr auto src1_args = TensorAccessorArgs<1>();
#endif

    // For sharded inputs: mark all tiles as available immediately (data already in L1)
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

    // Block/width sharded path: tiles are arranged in a 2D grid pattern
    if constexpr (block_or_width_sharded) {
        uint32_t row_start_tile_id = start_id;
        for (uint32_t h = 0; h < block_height; h++) {
            uint32_t tile_id = row_start_tile_id;
            for (uint32_t w = 0; w < block_width; w++) {
#ifndef IN0_SHARDED
                cb_reserve_back(cb_id_in0, onetile);  // Wait for space in CB
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(tile_id, s0, l1_write_addr_in0);  // DMA read from DRAM
#endif
#ifndef IN1_SHARDED
                cb_reserve_back(cb_id_in1, onetile);
                l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                noc_async_read_tile(tile_id, s1, l1_write_addr_in1);
#endif
                tile_id++;
                noc_async_read_barrier();  // Wait for DMA completion
#ifndef IN0_SHARDED
                cb_push_back(cb_id_in0, onetile);  // Make tile available to compute
#endif
#ifndef IN1_SHARDED
                cb_push_back(cb_id_in1, onetile);
#endif
            }
            // Stride across cores: jump by num_cores_y * block_width to next row
            row_start_tile_id += num_cores_y * block_width;
        }
    } else {
        // Simple linear tile iteration for height-sharded or interleaved inputs
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

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
(Alternative for block/width-sharded non-sharded output: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`)

#### Annotated Writer Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);   // DRAM address of output tensor
    const uint32_t num_pages = get_arg_val<uint32_t>(1);   // Number of tiles to write
    const uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting tile ID for this core

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Page size is read from the CB configuration at runtime
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    // For sharded output: just wait for compute to finish writing all tiles
    cb_wait_front(cb_id_out, num_pages);
#else
    constexpr uint32_t onepage = 1;
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onepage);    // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_page(i, s, l1_read_addr);  // DMA write to DRAM
        noc_async_writes_flushed();            // Ensure write is dispatched
        cb_pop_front(cb_id_out, onepage);      // Free CB slot for compute
    }
    noc_async_write_barrier();  // Wait for all writes to complete
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

This is a generic compute kernel shared by all binary SFPU operations. The specific operation is controlled entirely through compile-time defines injected by the program factory.

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
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"         // Provides binary_left_shift_tile, binary_shift_tile_init
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

// PRE_SCALE is true if either input requires a pre-processing SFPU step (e.g., type conversion).
// For LEFT_SHIFT, neither is defined, so PRE_SCALE is false.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime arguments: how many blocks to process and how many tiles per block
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // Input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // Input B circular buffer

    // For LEFT_SHIFT, no pre-processing is defined, so cb_inp0 == cb_in0 and cb_inp1 == cb_in1
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;  // Interim CB for pre-processed input A
#else
    constexpr auto cb_inp0 = cb_in0;             // Read directly from input A
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;  // Interim CB for pre-processed input B
#else
    constexpr auto cb_inp1 = cb_in1;             // Read directly from input B
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;  // Output circular buffer

    // Initialize the unpack/pack pipeline for tile copy operations
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // Main processing loop: iterate over blocks of tiles
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- PRE-PROCESSING PHASES (skipped for LEFT_SHIFT) ---
        // If SFPU_OP_INIT_PRE_IN0_0 were defined, input A tiles would be:
        //   1. Copied from cb_in0 to DST
        //   2. Transformed via SFPU_OP_FUNC_PRE_IN0_0
        //   3. Packed to cb_inp0 (interim buffer)
        // Same pattern for SFPU_OP_INIT_PRE_IN1_0 with input B -> cb_inp1.

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        // ... pre-processing for input A (not active for LEFT_SHIFT)
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
        // ... pre-processing for input B (not active for LEFT_SHIFT)
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

        // --- MAIN COMPUTE PHASE ---
        // Wait for both input tiles to be ready
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        // Acquire DST registers for SFPU computation
        tile_regs_acquire();
        tile_regs_wait();

        // Load input A tiles into even DST slots (0, 2, 4, ...)
        // copy_tile_to_dst_init_short_with_dt configures the unpacker for the
        // source CB format while keeping the dest CB for format conversion context.
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);  // CB tile i -> DST[i*2]
        }

        // Load input B tiles into odd DST slots (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);  // CB tile i -> DST[i*2+1]

            // For LEFT_SHIFT, the following define is active:
            //   SHIFT_INIT expands to: binary_shift_tile_init();
            //   BINARY_SFPU_OP expands to: binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2);
            //
            // This means: shift value at DST[i*2] left by amount at DST[i*2+1],
            // storing the result back into DST[i*2].
#ifdef BINOP_INIT
            BINOP_INIT
#endif
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
            SHIFT_INIT       // -> binary_shift_tile_init();
#endif

#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP   // -> binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2);
#endif
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            // Pack the result tile from DST[i*2] into the output CB
            pack_tile(i * 2, cb_out0);
        }
        tile_regs_commit();
        tile_regs_release();

        // Release input tiles and push output tiles
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### Call Chain

The full dispatch chain from the compute kernel to the SFPU microcode is:

1. `binary_left_shift_tile<DataFormat::Int32>(idst0, idst1, odst)` -- Compute API (binary_shift.h)
2. `llk_math_eltwise_binary_sfpu_left_shift<APPROX, DataFormat::Int32>(idst0, idst1, odst)` -- LLK dispatch (llk_math_eltwise_binary_sfpu_shift.h)
3. `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_binary_left_shift<...>, idst0, idst1, odst, VectorMode::RC)` -- LLK params framework (llk_math_eltwise_binary_sfpu_params.h)
4. `_calculate_binary_left_shift_<APPROX, 8, INT32, false>(idst0, idst1, odst)` -- SFPU microcode (ckernel_sfpu_shift.h)

#### SFPU Kernel File (Blackhole)

`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h`

#### SFPU Kernel File (Wormhole B0)

`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_shift.h`

#### Annotated SFPU Kernel Source (Blackhole)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_addrmod.h"
#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_binary_left_shift_(
    const std::uint32_t dst_index_in0,   // DST tile index for operand (value to shift)
    const std::uint32_t dst_index_in1,   // DST tile index for shift amount
    const std::uint32_t dst_index_out)   // DST tile index for result
{
    // Validate instruction mode at compile time.
    // INT32 is used for Int32/UInt32; LO16 for UInt16.
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE),
        "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Determine SFPLOAD/SFPSTORE format modifier:
    // SIGN_MAGNITUDE_FORMAT selects 2's complement conversion mode; otherwise use the
    // native instruction mode (INT32 for 32-bit types, LO16 for 16-bit).
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    // SFPU microcode: process 8 iterations per face (8 rows x 4 faces = 32 rows per tile face pair)
    // The outer framework (_llk_math_eltwise_binary_sfpu_params_) calls this function 4 times
    // for VectorMode::RC, advancing the dst_reg pointer between face pairs via SETRWC.
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile occupies 64 rows in the Destination register (32x32 tile = 4 faces x 16 rows,
        // but in Dest layout each face is 16 rows, so a tile is 64 rows total).
        constexpr std::uint32_t dst_tile_size = 64;

        // STEP 1: Load the value to shift from Dest into LREG0
        // ADDR_MOD_7 configures the address modifier for auto-increment behavior
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);

        // STEP 2: Load the shift amount from Dest into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);

        // STEP 3: Validate shift amount -- if < 0 or >= 32, result must be 0
        // SFPSETCC with mode 4: sets CC if LREG1 (shift_amount) is negative
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);

        // SFPIADD: add -32 (0xFE0 in 12-bit signed) to shift_amount, store in LREG2
        // Mode 1: update CC based on sign of result -- if shift_amount >= 32,
        // then shift_amount - 32 >= 0, so CC will be set accordingly
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);

        // SFPCOMPC: complement the CC -- this combines the two conditions:
        // after SFPSETCC (negative check) and SFPIADD (>= 32 check),
        // SFPCOMPC inverts CC so that lanes where shift_amount is invalid get CC=true
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // SFPMOV: for lanes where CC is true (invalid shift), move 0 (LCONST_0) into LREG0
        // This zeroes the result for out-of-range shift amounts
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // SFPENCC: disable conditional execution -- all subsequent instructions run unconditionally
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // STEP 4: Perform the actual left shift
        // SFPSHFT: shift LREG0 (value) left by the amount in LREG1 (shift_amount)
        // For lanes that were zeroed in step 3, shifting 0 still yields 0 (correct behavior)
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

        // STEP 5: Store the result from LREG0 back to the Dest register at the output tile index
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);

        // Advance the destination register row pointer for the next iteration
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### Annotated SFPU Kernel Source (Wormhole B0)

```cpp
// The Wormhole B0 implementation is functionally identical to Blackhole,
// with one key difference: ADDR_MOD_3 is used instead of ADDR_MOD_7
// for SFPLOAD and SFPSTORE instructions.

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_binary_left_shift_(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out)
{
    static_assert(
        is_valid_instruction_mode(INSTRUCTION_MODE),
        "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size = 64;
        // Wormhole uses ADDR_MOD_3 (different address modifier configuration)
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}
```

#### LLK Params Framework

**File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`

The params framework is responsible for calling the SFPU function the correct number of times to cover all four faces of a 32x32 tile:

```cpp
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    // Validate DST indices are within bounds
    LLK_ASSERT((dst_index_in0 < get_dest_max_tiles<...>()), "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT((dst_index_in1 < get_dest_max_tiles<...>()), "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT((dst_index_out < get_dest_max_tiles<...>()), "dst_index_out exceeds max dest tiles");

    // Signal the start of an SFPU operation (configures hardware state)
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0);

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::RC)
    {
        // Process all four faces of the tile (standard for elementwise operations)
        // Each face is 16 rows; the SFPU function processes 8 rows per call (ITERATIONS=8)
        // So 4 calls x 8 iterations = 32 row-groups, but SETRWC advances by 16 rows between calls
        for (int face = 0; face < 4; face++)
        {
            sfpu_func(dst_index_in0, dst_index_in1, dst_index_out, args...);
            // SETRWC advances the dest register row counter by 16 rows (8+8)
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    // VectorMode::R and VectorMode::C are also supported but not used for this operation

    // Signal completion of the SFPU operation
    _llk_math_eltwise_binary_sfpu_done_();
}
```

#### SFPU Instructions Used

| Instruction | Opcode | Description |
|-------------|--------|-------------|
| `SFPLOAD` | 0x70 | Loads a value from the Destination register file into an LREG. The `InstrMod` field selects the data format (INT32, LO16, etc.) and the `AddrMod` field controls address auto-increment. |
| `SFPSTORE` | 0x72 | Stores a value from an LREG back to the Destination register file. Same format and addressing options as SFPLOAD. |
| `SFPSETCC` | 0x7B | Sets the Condition Code (CC) register based on the value in RG[VC]. Mode 4 sets CC if the value is negative (sign bit check). |
| `SFPIADD` | 0x79 | Performs 2's complement integer addition between an immediate (Imm12) and RG[VC], storing to RG[VD]. Mode 1 updates CC based on the sign of the result. |
| `SFPCOMPC` | 0x8B | Conditionally complements (inverts) CC.Res. Used to combine multiple condition checks into a single conditional execution state. |
| `SFPMOV` | 0x7C | Moves RG[VC] to RG[VD]. Used here to zero out LREG0 (from LCONST_0) for invalid shift amounts. Only executes on lanes where CC is enabled. |
| `SFPENCC` | 0x8A | Directly sets CC.En and CC.Res. Used here to disable conditional execution (return to unconditional mode). |
| `SFPSHFT` | 0x7A | Performs a bitwise shift of RG[VD] by the amount in RG[VC]. Positive amounts shift left, negative amounts shift right. IPC=1, Latency=1. |
| `SETRWC` | N/A (control) | Sets the Read/Write Counter for the Destination register, used by the params framework to advance between tile faces. |

#### SFPU Register Usage

| Register | Usage in Left Shift |
|----------|-------------------|
| `LREG0` | Holds the value to be shifted (loaded from `dst_index_in0`). After the shift, holds the result. |
| `LREG1` | Holds the shift amount (loaded from `dst_index_in1`). Used as the shift source for SFPSHFT. |
| `LREG2` | Scratch register for the overflow check (`shift_amount - 32`). |
| `LCONST_0` | Fixed constant register containing 0. Used to zero out invalid results. |

Notably, the left shift kernel uses only 3 of the 8 available LREGs (plus one constant), making it one of the simpler SFPU operations. Compare this to the right shift kernel which additionally uses LREG3, LREG4 for sign-extension logic.

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` and `tile_regs_wait()` to gain exclusive access to the Dest register file. Both input tiles are copied from their CBs into adjacent Dest slots using `copy_tile` (operand A at even index, operand B at odd index).

2. **SFPU init**: `binary_shift_tile_init()` is called, which expands to `llk_math_eltwise_binary_sfpu_shift_init<APPROX>()`. This calls the generic `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>()` to configure the SFPU hardware for binary operation mode.

3. **SFPU dispatch**: `binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2)` is called. The template parameter determines:
   - `INSTRUCTION_MODE = InstrModLoadStore::INT32` (for Int32/UInt32)
   - `SIGN_MAGNITUDE_FORMAT = false` (default)

4. **Per-face iteration**: The LLK params framework calls `_calculate_binary_left_shift_` 4 times (once per tile face in RC mode). Between each call, `SETRWC` advances the Dest register row pointer by 16 rows.

5. **Per-row iteration**: Within each face, the SFPU function loops 8 times (ITERATIONS=8), processing one row per iteration:
   - **Load**: SFPLOAD reads one row from each of the two input tiles in Dest into LREG0 (value) and LREG1 (shift amount).
   - **Validate**: A 3-instruction sequence (SFPSETCC + SFPIADD + SFPCOMPC) checks if shift_amount < 0 or shift_amount >= 32.
   - **Zero invalid**: SFPMOV conditionally writes 0 into LREG0 for lanes with invalid shift amounts.
   - **Clear CC**: SFPENCC returns to unconditional execution mode.
   - **Shift**: SFPSHFT performs `LREG0 = LREG0 << LREG1` on all lanes.
   - **Store**: SFPSTORE writes LREG0 back to the output tile location in Dest.
   - **Advance**: `dst_reg++` increments the row pointer.

6. **Pack**: After the SFPU completes, `pack_tile(i*2, cb_out0)` moves the result from Dest to the output CB. The packer converts from Dest format back to the output tensor's data format.

7. **Release**: `tile_regs_commit()` and `tile_regs_release()` release the Dest registers. `cb_pop_front` frees input CB slots, `cb_push_back` publishes output tiles.

#### SFPU Configuration

| Configuration | Value | Notes |
|--------------|-------|-------|
| `APPROXIMATION_MODE` | Determined by `APPROX` macro | Not meaningfully used for integer shift (no approximation involved) |
| `ITERATIONS` | 8 | Processes 8 rows per SFPU function call (one 16-row face requires 2 SETRWC advances of 8 rows each) |
| `INSTRUCTION_MODE` | `InstrModLoadStore::INT32` for Int32/UInt32; `InstrModLoadStore::LO16` for UInt16 | Controls SFPLOAD/SFPSTORE format interpretation |
| `SIGN_MAGNITUDE_FORMAT` | `false` | When true, would use INT32_2S_COMP mode for sign-magnitude to 2's complement conversion |
| `fp32_dest_acc_en` | `true` | Always enabled for Int32/UInt32 output to use full 32-bit Dest width |
| `unpack_to_dest_mode` | `UnpackToDestFp32` | Bypasses FPU, loads data directly to Dest for SFPU processing |

#### Hardware Compatibility Notes

The Blackhole and Wormhole B0 implementations of `_calculate_binary_left_shift_` are **functionally identical** with one difference:

| Aspect | Wormhole B0 | Blackhole |
|--------|-------------|-----------|
| Address Modifier | `ADDR_MOD_3` | `ADDR_MOD_7` |
| Include | `ckernel_ops.h`, `sfpi.h` | `ckernel_addrmod.h`, `ckernel_ops.h`, `sfpi.h` |

The `ADDR_MOD` difference reflects different address modifier register configurations between the two architectures. The address modifier controls how the SFPLOAD/SFPSTORE address is computed (base + offset with auto-increment). The SFPU instruction behavior (SFPSHFT, SFPSETCC, etc.) is identical across both architectures for this operation.

The SFPSHFT instruction itself (opcode 0x7A) has the same semantics on both Wormhole and Blackhole:
- Encoding: O2 format
- Input/Output: INT32
- IPC: 1 instruction per cycle
- Latency: 1 cycle
- Positive shift amounts = left shift, negative = right shift
- `InstrMod[0]` selects between register and immediate shift source
- `InstrMod[1]` selects logical vs arithmetic shift (relevant only for right shifts)
- `InstrMod[2]` selects the data source (RG[VD] or RG[VC])

---

## Runtime Arguments

### Reader Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `src0_addr` | DRAM base address of input tensor A |
| 1 | `src1_addr` | DRAM base address of input tensor B |
| 2 | `num_tiles` | Number of tiles assigned to this core |
| 3 | `start_id` | Starting tile ID for this core's work partition |
| 4 | `block_height` | Block height (for block/width sharded layout) |
| 5 | `block_width` | Block width (for block/width sharded layout) |
| 6 | `num_cores_y` | Number of cores in Y dimension (stride computation) |

### Writer Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `dst_addr` | DRAM base address of output tensor |
| 1 | `num_pages` | Number of tiles to write |
| 2 | `start_id` | Starting tile ID for this core |

### Compute Kernel Runtime Args

| Index | Name | Description |
|-------|------|-------------|
| 0 | `per_core_block_cnt` | Number of blocks to process |
| 1 | `per_core_block_size` | Number of tiles per block |

---

## Work Distribution

Work is distributed across cores via `set_eltwise_binary_runtime_args`, defined in `eltwise_multi_core_program_factory_common.hpp`. The total number of tiles is divided among available cores, with some cores potentially processing one extra tile to handle remainders. The `worker_grid` from `operation_attributes` defines which cores participate.

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Element-wise multi-core SFPU program factory architecture, compute kernel dispatch patterns, program factory selection logic.
- **tenstorrent/tt-llk**: `ckernel::sfpu::_calculate_binary_left_shift_` implementation details, LLK API function chain (`llk_math_eltwise_binary_sfpu_left_shift` -> `_llk_math_eltwise_binary_sfpu_params_`), SFPU instruction usage in shift operations, ADDR_MOD differences between Wormhole and Blackhole.
- **tenstorrent/sfpi**: SFPI operator overloads for left shift (`operator<<` on `vInt`/`vUInt`), underlying `__builtin_rvtt_sfpshft_i` and `__builtin_rvtt_sfpshft_v` intrinsics, logical vs arithmetic shift mode control.
- **tenstorrent/tt-isa-documentation**: Not directly queried (Confluence provided sufficient SFPU ISA detail).

### Confluence References

- **Tensix SFPU Instruction Set Architecture** (Page ID: 1170505767): Consulted for authoritative documentation of the following instructions used in the left shift kernel:
  - **SFPSHFT** (0x7A): Bitwise shift instruction -- positive amounts shift left, negative shift right. O2 encoding, INT32 format, 1 IPC, 1 cycle latency.
  - **SFPLOAD** (0x70): Dest-to-LREG load with format conversion.
  - **SFPSTORE** (0x72): LREG-to-Dest store with format conversion.
  - **SFPSETCC** (0x7B): Condition code setting based on register value.
  - **SFPIADD** (0x79): Integer add/subtract with immediate and CC update.
  - **SFPCOMPC** (0x8B): Conditional CC complement for combining conditions.
  - **SFPMOV** (0x7C): Register move (conditional on CC).
  - **SFPENCC** (0x8A): CC enable/disable control.

### Glean References

Not consulted. DeepWiki and Confluence provided sufficient detail for this analysis.
