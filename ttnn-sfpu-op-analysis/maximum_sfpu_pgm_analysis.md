# SFPU Operation Analysis: MAXIMUM (Legacy Element-Wise Binary)

## Operation Overview

**Operation**: MAXIMUM (binary element-wise)
**Program Factory**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`
**Namespace**: `ttnn::operations::binary`
**Factory Class**: `BinaryDeviceOperation::ElementWiseMultiCoreSfpu`

The MAXIMUM operation computes `y = max(x0, x1)` element-wise across two input tensors. It is implemented as an SFPU operation using the `SFPSWAP` hardware instruction for comparison and swap, which is significantly more efficient than the alternative FPU-based subtract-then-compare approach.

### When This Factory is Selected

The `ElementWiseMultiCoreSfpu` factory is selected by `BinaryDeviceOperation::select_program_factory` when:
1. Both input tensors have the same shape (height_a == height_b and width_a == width_b), AND
2. `utils::is_binary_sfpu_op(op, dtype1, dtype2)` returns `true`

For `BinaryOpType::MAXIMUM`, `is_binary_sfpu_op` returns `true` unconditionally (for all data type combinations). This means MAXIMUM always routes through the SFPU path when tensors have matching shapes.

Source: `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` lines 58-62.

### Data Type Variants

The MAXIMUM operation has three SFPU kernel variants selected at compile time via preprocessor defines:

| Input Types | Init Define (`BINOP_INIT`) | Op Define (`BINARY_SFPU_OP`) | SFPU Function |
|---|---|---|---|
| Float (default) | `binary_max_tile_init();` | `binary_max_tile(i*2, i*2+1, i*2);` | `calculate_binary_max_min<true>` |
| INT32 + INT32 | `binary_max_int32_tile_init();` | `binary_max_int32_tile(i*2, i*2+1, i*2);` | `calculate_binary_max_min_int32<true, false>` |
| UINT32 + UINT32 | `binary_max_uint32_tile_init();` | `binary_max_uint32_tile(i*2, i*2+1, i*2);` | `calculate_binary_max_min_int32<true, true>` |

Source: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` lines 334-344.

---

## Program Structure

### Circular Buffer Configuration

| CB Index | Name | Purpose | Size (non-sharded) | Size (sharded) |
|---|---|---|---|---|
| `c_0` | `cb_src0` | Input tensor A | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_1` | `cb_src1` | Input tensor B | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_2` | `cb_out0` | Output tensor | `2 * max_block_size * tile_size` | `num_tiles_per_shard * tile_size` |
| `c_3` | `cb_inp0` | Interim for pre-scaled input A | `max_block_size * tile_size` (only if `SFPU_OP_INIT_PRE_IN0_0` defined) | Same |
| `c_4` | `cb_inp1` | Interim for pre-scaled input B | `max_block_size * tile_size` (only if `SFPU_OP_INIT_PRE_IN1_0` defined) | Same |

For the MAXIMUM operation specifically, **no prescaling is needed**, so `c_3` and `c_4` are not allocated. The `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` defines are not set for MAXIMUM.

The `max_block_size` is computed as the largest power-of-2 factor of `num_tiles_per_shard` (for sharded inputs) or defaults to 1 (for interleaved inputs). For interleaved inputs, each CB holds 2 tiles (double-buffered).

### Sharding Support

The factory supports three sharding configurations:
- **Input A sharded**: `IN0_SHARDED` define set, CB `c_0` uses globally allocated address from input buffer
- **Input B sharded**: `IN1_SHARDED` define set, CB `c_1` uses globally allocated address from input buffer
- **Output sharded**: `OUT_SHARDED` define set, CB `c_2` uses globally allocated address from output buffer
- **Block/width sharded**: Special writer kernel selected for non-height-sharded layouts

### FP32 Destination Accumulation

For MAXIMUM (which is not `BinaryOpType::POWER`), `UnpackToDestMode::UnpackToDestFp32` is set for all input CBs (`c_0`, `c_1`, `c_3`, `c_4`). This means data is unpacked directly to the destination register in FP32 format, regardless of the input tensor's data format. The `fp32_dest_acc_en` flag is set when the output data format is Float32, Int32, or UInt32.

---

## Kernel Implementations

### Reader Kernel

**Path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`

**Compile-time args**:
- `arg[0]`: `block_or_width_sharded` (bool)
- Followed by `TensorAccessorArgs` for src0 (if not sharded) and src1 (if not sharded)

**Runtime args**:
- `arg[0]`: `src0_addr` - DRAM address of input tensor A
- `arg[1]`: `src1_addr` - DRAM address of input tensor B
- `arg[2]`: `num_tiles` - total number of tiles to process
- `arg[3]`: `start_id` - starting tile index for this core
- `arg[4]`: `block_height` - height of shard block (for block/width sharded)
- `arg[5]`: `block_width` - width of shard block (for block/width sharded)
- `arg[6]`: `num_cores_y` - number of cores in Y dimension (for stride calculation)

**Behavior**: For sharded inputs, simply reserves and pushes back the entire shard (data is already in L1). For interleaved inputs, reads tiles one at a time from DRAM via NoC, using `TensorAccessor` for address translation. Supports two iteration patterns: block/width sharded (2D iteration with stride) and standard (linear iteration).

#### Annotated Reader Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime arguments: addresses, tile count, start offset, and sharding geometry
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t block_height = get_arg_val<uint32_t>(4);
    uint32_t block_width = get_arg_val<uint32_t>(5);
    uint32_t num_cores_y = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    // Compile-time arg: whether the tensor is block or width sharded
    constexpr bool block_or_width_sharded = get_compile_time_arg_val(0) == 1;

    // TensorAccessor compile-time args are chained: src0 first, then src1
#if !defined(IN0_SHARDED) && !defined(IN1_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
#elif !defined(IN0_SHARDED)
    constexpr auto src0_args = TensorAccessorArgs<1>();
#elif !defined(IN1_SHARDED)
    constexpr auto src1_args = TensorAccessorArgs<1>();
#endif

    // For sharded inputs, data is already in L1 at the CB's globally allocated address.
    // Just signal that all tiles are available.
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
        // Block/width sharded iteration: 2D loop over block height x width
        // Stride between rows is num_cores_y * block_width (tiles are distributed across cores)
        uint32_t row_start_tile_id = start_id;
        for (uint32_t h = 0; h < block_height; h++) {
            uint32_t tile_id = row_start_tile_id;
            for (uint32_t w = 0; w < block_width; w++) {
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
                tile_id++;
                noc_async_read_barrier(); // Wait for NoC read to complete before signaling availability
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
        // Standard linear iteration: one tile at a time from start_id
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

**Path** (standard): `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
**Path** (block/width sharded, non-sharded output): `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`

**Compile-time args**:
- `arg[0]`: `output_cb_index` (always `c_2`)
- Followed by `TensorAccessorArgs` for the output buffer

The writer kernel selection depends on sharding configuration. When the input is block or width sharded but the output is not sharded, a special writer that handles the block-to-interleaved conversion is used.

### Compute Kernel

**Path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

**Compile-time configuration** (via defines from `get_defines_fp32`):
- `BINOP_INIT`: `binary_max_tile_init();` (for float), `binary_max_int32_tile_init();` (for int32), `binary_max_uint32_tile_init();` (for uint32)
- `BINARY_SFPU_OP`: `binary_max_tile(i*2, i*2+1, i*2);` (or int32/uint32 variants)
- `fp32_dest_acc_en`: Enabled for Float32/Int32/UInt32 output formats
- `unpack_to_dest_mode`: `UnpackToDestFp32` for all input CBs (since MAXIMUM is not POWER)

**Runtime args**:
- `arg[0]`: `per_core_block_cnt` - number of blocks to process
- `arg[1]`: `per_core_block_size` - number of tiles per block

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
#include "api/compute/binary_shift.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/remainder_int32.h"
#include "api/compute/binary_fmod.h"
// This header provides binary_max_tile, binary_max_int32_tile, binary_max_uint32_tile
// and their _init counterparts, which dispatch to the SFPU max/min kernel
#include "api/compute/binary_max_min.h"
#include "api/compute/xlogy.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// PRE_SCALE is true if either input needs prescaling (e.g., for LOGADDEXP where exp() is applied first).
// For MAXIMUM, neither SFPU_OP_INIT_PRE_IN0_0 nor SFPU_OP_INIT_PRE_IN1_0 is defined, so PRE_SCALE is false.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    // Runtime args: how many blocks and how many tiles per block this core processes
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // Input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // Input B circular buffer

    // If prescaling is defined for input A, use interim CB c_3; otherwise use c_0 directly.
    // For MAXIMUM, cb_inp0 == cb_in0 (no prescaling).
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;
#else
    constexpr auto cb_inp0 = cb_in0;
#endif

    // Same for input B: for MAXIMUM, cb_inp1 == cb_in1
#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;
#else
    constexpr auto cb_inp1 = cb_in1;
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;  // Output circular buffer

    // Initialize the unary op common state (sets up packer, unpacker base configuration)
    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    // Main processing loop: iterate over blocks
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // [PRESCALING SECTIONS - NOT ACTIVE FOR MAXIMUM]
        // SFPU_OP_INIT_PRE_IN0_0 / SFPU_OP_INIT_PRE_IN1_0 blocks would apply
        // element-wise transforms (e.g., exp()) to inputs before the binary op.
        // For MAXIMUM, these are compiled out.

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        // ... prescaling for input A (not active for MAXIMUM) ...
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
        // ... prescaling for input B (not active for MAXIMUM) ...
#endif

        // Wait for both inputs to be available in their respective CBs
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        // Reserve space in output CB for the results
        cb_reserve_back(cb_out0, per_core_block_size);

        // Acquire destination registers for SFPU computation
        tile_regs_acquire();
        tile_regs_wait();

        // Copy input A tiles from CB c_0 into DST at even indices (0, 2, 4, ...)
        // The copy_tile_to_dst_init_short_with_dt call configures the unpacker
        // for the data type transition from cb_inp1's format to cb_inp0's format
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);  // CB tile i -> DST[i*2]
        }

        // Copy input B tiles from CB c_1 into DST at odd indices (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);  // CB tile i -> DST[i*2+1]

            // For MAXIMUM, BINOP_INIT expands to binary_max_tile_init() (or int32/uint32 variant).
            // This configures the SFPU for the swap-based max operation, including
            // setting up SFPLOADMACRO instruction templates and address modifiers.
#ifdef BINOP_INIT
            BINOP_INIT  // binary_max_tile_init(); -- configures SFPU for max comparison
#endif
            // ... other init macros for other op types (not active for MAXIMUM) ...

            // BINARY_SFPU_OP expands to binary_max_tile(i*2, i*2+1, i*2);
            // This executes the SFPU max operation:
            //   - Reads from DST[i*2] (input A) and DST[i*2+1] (input B)
            //   - Computes element-wise max using SFPSWAP instruction
            //   - Writes result back to DST[i*2] (overwrites input A location)
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP  // binary_max_tile(i*2, i*2+1, i*2);
#endif

            // ... other op chain macros (not active for MAXIMUM) ...

            // Pack the result tile from DST[i*2] to the output CB
            pack_tile(i * 2, cb_out0);
        }

        // Commit destination registers (signal packer that data is ready)
        tile_regs_commit();
        // Release destination registers for next iteration
        tile_regs_release();

        // Free input tiles and publish output tiles
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### Compute API Layer

**Path**: `tt_metal/hw/inc/api/compute/binary_max_min.h`

The compute API header provides three tile-level functions for MAXIMUM:

- `binary_max_tile(idst0, idst1, odst, vector_mode)` -- floating-point max
- `binary_max_int32_tile(idst0, idst1, odst)` -- signed int32 max
- `binary_max_uint32_tile(idst0, idst1, odst)` -- unsigned int32 max

Each wraps a call to the `MATH(())` macro (which executes on TRISC_MATH), dispatching to the LLK layer.

```cpp
// From tt_metal/hw/inc/api/compute/binary_max_min.h

// Float max: uses SFPSWAP with VEC_MIN_MAX mode directly
ALWI void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    MATH((llk_math_eltwise_binary_sfpu_binary_max<APPROX>(idst0, idst1, odst, vector_mode)));
}
ALWI void binary_max_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binary_max_init<APPROX>())); }

// Int32 max: uses SFPSWAP + SFPSETCC correction for sign-magnitude
ALWI void binary_max_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binary_max_int32<APPROX>(idst0, idst1, odst)));
}
ALWI void binary_max_int32_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binary_max_int32_init<APPROX>())); }

// Uint32 max: uses SFPSWAP with inverted comparison + GTE0 correction
ALWI void binary_max_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_binary_max_uint32<APPROX>(idst0, idst1, odst)));
}
ALWI void binary_max_uint32_tile_init() { MATH((llk_math_eltwise_binary_sfpu_binary_max_uint32_init<APPROX>())); }
```

#### LLK Math Layer

**Path (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h`
**Path (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h`

Both architectures share identical LLK layer code. The LLK layer:
1. Calls `llk_math_eltwise_binary_sfpu_init<SfpuType::max, APPROXIMATE>` with the init function
2. Calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>` with the calculate function

The init function (`_llk_math_eltwise_binary_sfpu_init_`) configures SFPU address modifiers and the SFPU configuration register. The params function iterates over tile faces according to `vector_mode` (Row, Column, or RC) and calls the SFPU calculate function for each face.

```cpp
// From llk_math_eltwise_binary_sfpu_max_min.h (identical for Wormhole and Blackhole)

// Float maximum
template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max, APPROXIMATE>(sfpu::binary_max_min_init<true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_max_min<true>, dst_index0, dst_index1, odst, vector_mode);
}

// Int32 maximum
template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max_int32, APPROXIMATE>(sfpu::binary_max_min_int32_init<true, false>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_max_min_int32<true, false>, dst_index0, dst_index1, odst, vector_mode);
}

// Uint32 maximum
template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max_uint32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::max_uint32, APPROXIMATE>(sfpu::binary_max_min_int32_init<true, true>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_binary_max_uint32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_max_min_int32<true, true>, dst_index0, dst_index1, odst, vector_mode);
}
```

#### SFPU Kernel File

**Path (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`
**Path (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`

#### Annotated SFPU Kernel Source (Wormhole)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2026 Jason Davies <jason@jasondavies.com>
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ============================================================================
// FLOATING-POINT MAX/MIN KERNEL
// ============================================================================
// This function computes element-wise max (IS_MAX_OP=true) or min (IS_MAX_OP=false)
// between two tiles already loaded into the DST register file.
// It processes ITERATIONS rows (default 8 = 4 faces x 2 rows per face half,
// but the actual face iteration is handled by the LLK params wrapper).
//
// The core operation is SFPSWAP with VEC_MIN_MAX mode, which atomically compares
// two LREG values across all 32 SIMD lanes and places min in one register and
// max in the other.
template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Convert tile indices to byte offsets in the DST register file.
    // Each tile occupies 32 rows, and the offset is doubled because DST uses
    // 16-bit addressing (each row = 2 units of 16 bits in the address space).
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
    // FALLBACK PATH: Simple sequential load-compare-store (slower, ~4 cycles per row)
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row from input A into LREG0 (32 elements across SIMD lanes)
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset0);
        // Load one row from input B into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset1);
        // SFPSWAP with VEC_MIN_MAX mode: after execution,
        //   LREG0 = min(original_LREG0, original_LREG1) across all 32 lanes
        //   LREG1 = max(original_LREG0, original_LREG1) across all 32 lanes
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        // Store the desired result: LREG1 (max) for IS_MAX_OP=true, LREG0 (min) otherwise
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset2);
    }
#else
    // OPTIMIZED PATH: Uses SFPLOADMACRO for pipelined execution (3 cycles per row)
    //
    // SFPLOADMACRO allows scheduling loads, SFPU operations, and stores in a
    // pipelined manner across the Load, Simple, MAD, Round, and Store sub-units
    // of the SFPU. This achieves ~1.33x speedup over the sequential path.
    //
    // Pipeline schedule (repeating pattern for each row):
    //
    // t | Load | Simple              | MAD | Round     | Store   |
    // - | ---- | ------------------- | --- | --------- | ------- |
    // 0 | [a]  |                     |     |           |         |
    // 1 |  b   |                     |     |           |         |
    // 2 | [c]  | swap_minmax([a], b) |     |           |         |
    // 0 | ...  |                     |     |           |         |
    // 1 | ...  |                     |     | L16 = [a] |         |
    // 2 | ...  |                     |     |           | [c] L16 |
    //
    // [x] = scheduled by SFPLOADMACRO with VD=x
    // Macro 0 handles: load a, execute swap_minmax template, round (convert to L16 for store)
    // Macro 1 handles: store result c

    constexpr int b = p_sfpu::LREG2;   // Register for input B row
    constexpr int c = p_sfpu::LREG3;   // Register used for store scheduling

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // Alternate between LREG0 and LREG1 to avoid pipeline stalls
        // Macro 0: load input A row into register 'a', triggers swap template on Simple sub-unit
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset0 | (a >> 2));
        // Standard load: input B row into register 'b' (LREG2)
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset1);
        // Macro 1: store result from register 'c', using Round sub-unit for format conversion
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset2 | (c >> 2));
    }

    // Pipeline drain: 3 NOPs to flush remaining in-flight operations
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

// ============================================================================
// INTEGER MAX/MIN KERNEL (INT32 and UINT32)
// ============================================================================
// Integer max/min requires extra steps because SFPSWAP operates on sign-magnitude
// representation (the hardware's native float format), which does not correctly
// order two's-complement integers. After the SFPSWAP, a correction step using
// SFPSETCC + conditional SFPSWAP fixes cases where the sign-magnitude comparison
// gave the wrong answer.
//
// For UNSIGNED integers (IS_UNSIGNED=true):
//   - SFPSWAP mod1=9 is used (which inverts the comparison)
//   - SFPSETCC checks GTE0 (treats high bit as unsigned magnitude)
//
// For SIGNED integers (IS_UNSIGNED=false):
//   - SFPSWAP with VEC_MIN_MAX mode
//   - SFPSETCC checks LT0 (detects negative numbers that were mis-ordered)
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
    // FALLBACK PATH: Sequential with correction (~7 cycles per row)
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load both inputs as INT32 (InstrModLoadStore::INT32 tells the load unit
        // to interpret the DST data as integer, not float)
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, offset1);

        // For unsigned: mod1=9 inverts comparison direction
        // For signed: VEC_MIN_MAX gives sign-magnitude ordering
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // CORRECTION STEP: SFPSWAP uses sign-magnitude comparison which can give
        // wrong results for two's complement integers. For example, -1 (0xFFFFFFFF)
        // would be treated as a large negative in sign-magnitude but is actually the
        // largest negative value in two's complement.
        //
        // The correction uses per-lane condition codes:
        //   For signed: set CC if LREG < 0 (the negative values need fixing)
        //   For unsigned: set CC if LREG >= 0 (high bit clear = needs no fix)
        // Then conditionally swap to correct the ordering.
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        // Conditional swap: only swaps lanes where the condition code is set
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        // Disable condition codes (return to unconditional execution)
        TTI_SFPENCC(0, 0, 0, 0);

        // Store the correct result as INT32
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, offset2);
    }
#else
    // OPTIMIZED PATH: SFPLOADMACRO pipelined execution (5 cycles per row)
    // Processes two rows per macro replay iteration (a0/b0 and a1/b1),
    // interleaving the correction steps within the pipeline.
    //
    // Pipeline schedule:
    //
    // t | Load | Simple                | MAD | Round     | Store   |
    // - | ---- | --------------------- | --- | --------- | ------- |
    // 0 | [a0] |                       |     |           |         |
    // 1 | [b0] |                       |     |           |         |
    // 2 |      | setcc  a1  (<0 or >=0)|     |           |         |
    // 3 |      | encc                  |     |           |         |
    // 4 | [c]  | swap_minmax([a0], b0) |     |           |         |
    // 0 | ...  |                       |     |           |         |
    // 1 | ...  | setcc [b0] (<0 or >=0)|     | L16 = [a] |         |
    // 2 | ...  | setcc  a0  (<0 or >=0)|     |           |         |
    // 3 | ...  | encc                  |     | L16 = [b] |         |
    // 4 | ...  |                       |     |           | [c] L16 |

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;

    // Record the instruction sequence into the replay buffer (10 instructions per pair of rows)
    lltt::record<lltt::NoExec>(0, 10);

    // First row of the pair: load a0, b0, apply correction for previous iteration's a1, store previous c
    TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a0 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b0 >> 2));
    TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

    // Second row of the pair: load a1, b1, apply correction for a0, store c
    TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a1 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b1 >> 2));
    TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

    // Replay the recorded sequence ITERATIONS/2 times (processes 2 rows per replay)
#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);
    }

    // Handle odd iteration count and pipeline drain
    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);     // One more half-iteration
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2); // Drain correction steps
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);     // Drain correction steps
    }

    TTI_SFPNOP;  // Final pipeline drain
#endif
}

// ============================================================================
// INITIALIZATION FUNCTIONS (configure SFPLOADMACRO templates)
// ============================================================================

// Float max/min init: configures the SFPLOADMACRO pipeline for 3-cycle throughput
template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]: SFPSWAP on the Simple sub-unit
    // mod1=9 means "VD gets max, VC gets min" (inverted from default VEC_MIN_MAX)
    // For IS_MAX_OP=true: mod1=9, so the LREG that was loaded by SFPLOADMACRO (VD)
    //   gets the max value, which is what we want to store.
    // For IS_MAX_OP=false (min): mod1=SFPSWAP_MOD1_VEC_MIN_MAX (default), so VD gets min.
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[1]: SFPSHFT2 for rounding/format conversion on the Round sub-unit
    TTI_SFPSHFT2(0, 0, 13, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0 configuration: Load -> Simple(swap) -> Round(convert) pipeline
    {
        // simple_bits: enabled, template 0 (swap), trigger on instruction 4, execute at step 1
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0;  // MAD sub-unit not used
        // round_bits: enabled, uses L16 format, template 1 (shft2), at step 3 using reg 5
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;
        constexpr uint store_bits = 0;  // Store handled by Macro 1

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);  // Write to Macro slot 0
    }

    // Macro 1 configuration: Store pipeline
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        // store_bits: enabled with L16 source, at step 2 using reg 3
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);  // Write to Macro slot 1
    }

    // Misc configuration:
    //   StoreMod0: DEFAULT (standard store format)
    //   UsesLoadMod0ForStore: {1,1} (both macros use load mod 0 for store addressing)
    //   UnitDelayKind: {1,1} (WaitForElapsedInstructions=1, ensures pipeline ordering)
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}

// Int32 max/min init: configures the SFPLOADMACRO pipeline for 5-cycle throughput
// (more complex due to the correction step using SFPSETCC + conditional SFPSWAP)
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int b1 = p_sfpu::LREG3;

    // InstructionTemplate[0]: SFPSWAP for first row of pair
    // XOR of IS_MAX_OP and IS_UNSIGNED determines comparison direction:
    //   MAX + signed:   mod1=9 (VD=max, VC=min)
    //   MAX + unsigned: mod1=VEC_MIN_MAX (inverted because unsigned needs correction anyway)
    //   MIN + signed:   mod1=VEC_MIN_MAX
    //   MIN + unsigned: mod1=9
    TTI_SFPSWAP(0, b0, 12, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[1]: SFPSWAP for second row of pair
    TTI_SFPSWAP(0, b1, 13, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[2]: SFPSETCC for correction
    TTI_SFPSETCC(0, 0, 14, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);

    // InstructionTemplate[3]: SFPSHFT2 for rounding/format conversion
    TTI_SFPSHFT2(0, 0, 15, 6);

    // Macro 0: Load a0, trigger swap template 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: Load a1, trigger swap template 1
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2: Load b, trigger setcc template 2
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (4 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Macro 3: Store result
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc configuration for 4 macros
    TTI_SFPCONFIG(0xff0, 8, 1);
#endif
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description | Usage in MAXIMUM |
|---|---|---|
| `SFPLOAD` | Load a row (32 elements) from DST register file into an SFPU local register (LREG) | Loads input A and input B rows into LREG0 and LREG1 |
| `SFPSWAP` (VEC_MIN_MAX) | Compare-and-swap across 32 SIMD lanes: places min in VC, max in VD | Core operation: performs the element-wise max comparison |
| `SFPSWAP` (mod1=9) | Inverted VEC_MIN_MAX: places max in VD, min in VC | Used in SFPLOADMACRO init for IS_MAX_OP=true to get max in the loaded register |
| `SFPSWAP` (SWAP) | Conditional swap based on per-lane condition codes | INT32 correction: conditionally swaps elements where sign-magnitude comparison was wrong |
| `SFPSTORE` | Store a row from an SFPU local register back to the DST register file | Writes the max result back to DST at the output tile offset |
| `SFPLOADMACRO` | Macro-scheduled load that triggers pipelined Simple/MAD/Round/Store operations | Optimized path: achieves 3-cycle (float) or 5-cycle (int32) throughput per row |
| `SFPSETCC` | Set per-lane condition codes based on register comparison | INT32 correction: flags lanes where values are negative (signed) or non-negative (unsigned) |
| `SFPENCC` | Enable/disable conditional execution based on condition codes | INT32: disables condition codes after correction swap |
| `SFPSHFT2` | Shift operation used for rounding/format conversion in pipeline | SFPLOADMACRO pipeline: converts internal format to L16 for store |
| `SFPLOADI` | Load immediate value into LREG (used for SFPLOADMACRO configuration) | Init functions: configures macro simple/mad/round/store bit fields |
| `SFPCONFIG` | Configure SFPU control registers (macros, misc settings) | Init functions: writes macro definitions and misc configuration |
| `SFPNOP` | No-operation (pipeline bubble) | Pipeline drain at end of SFPLOADMACRO sequences |

#### SFPU Register Usage

**Local Registers (LREGs)**:

| Register | Float Path | Int32 Path |
|---|---|---|
| `LREG0` | Input A row (alternates with LREG1 in SFPLOADMACRO path) | `a0` - first input A row |
| `LREG1` | Input A row (alternates with LREG0 in SFPLOADMACRO path) | `b0` - first input B row |
| `LREG2` | `b` - Input B row | `a1` - second input A row |
| `LREG3` | `c` - Store scheduling register | `b1` - second input B row |
| `LREG7` | Not used | `c` - Store scheduling register |

**DST Register File**:
- Input A tile: at `dst_index_in0 * 32` rows (= `i*2` tile slots in the compute kernel)
- Input B tile: at `dst_index_in1 * 32` rows (= `i*2+1` tile slots)
- Output tile: at `dst_index_out * 32` rows (= `i*2`, same as input A -- in-place overwrite)

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` and `tile_regs_wait()` to obtain exclusive access to the DST register buffer.

2. **Unpack input A**: `copy_tile(cb_inp0, i, i * 2)` unpacks tile `i` from circular buffer `c_0` into DST slot `i*2`. The unpacker is configured with `UnpackToDestFp32` mode, so data is converted to FP32 in DST regardless of the source format.

3. **Unpack input B**: `copy_tile(cb_inp1, i, i * 2 + 1)` unpacks tile `i` from circular buffer `c_1` into DST slot `i*2+1`.

4. **SFPU initialization**: `binary_max_tile_init()` (expanded from `BINOP_INIT`) configures the SFPU:
   - For the SFPLOADMACRO path: sets up instruction templates (SFPSWAP, SFPSHFT2), macro definitions (scheduling bits for Simple/MAD/Round/Store sub-units), and misc configuration.
   - For the DISABLE_SFPLOADMACRO path: no special init needed (the fallback path uses explicit instructions).

5. **SFPU max computation**: `binary_max_tile(i*2, i*2+1, i*2)` (expanded from `BINARY_SFPU_OP`) calls through the API/LLK chain to `calculate_binary_max_min<true>`:
   - The `_llk_math_eltwise_binary_sfpu_params_` wrapper handles face iteration based on `vector_mode` (default `RC` = all 4 faces) and calls the SFPU function for each face.
   - For each row within the face (8 iterations by default): loads one row each from the two input tile locations, performs SFPSWAP to compute max, stores the max row to the output tile location.
   - The SFPLOADMACRO path pipelines these operations to achieve 3 cycles per row (versus ~4 cycles sequential).

6. **Pack result**: `pack_tile(i * 2, cb_out0)` packs the result from DST slot `i*2` to the output circular buffer `c_2`.

7. **Cleanup**: `tile_regs_commit()` signals the packer, `tile_regs_release()` frees the DST buffer, `cb_pop_front()` frees input CB pages, `cb_push_back()` publishes output CB pages.

#### SFPU Configuration

- **Math fidelity**: Not directly applicable -- MAXIMUM does not use the FPU matrix engine. The `APPROX` template parameter is passed through but does not affect the SFPSWAP instruction behavior.
- **Unpack to dest mode**: `UnpackToDestFp32` for all input CBs (configures the unpacker to convert to FP32 in DST). This is set for all non-POWER binary SFPU operations.
- **fp32_dest_acc_en**: Set to `true` when output is Float32, Int32, or UInt32. Controls whether the DST register operates in 32-bit or 16-bit mode.
- **DISABLE_SFPLOADMACRO**: A compile-time define that, if set, disables the optimized SFPLOADMACRO pipeline and uses the simpler sequential path. This is useful for debugging or when SFPLOADMACRO has known issues on certain hardware revisions.

#### Hardware Compatibility Notes

**Wormhole vs Blackhole differences in the SFPU kernel**:

The `ckernel_sfpu_binary_max_min.h` source files for Wormhole and Blackhole are **nearly identical** in structure and logic. The key differences are:

1. **Address modifiers**: Wormhole uses `ADDR_MOD_3` / `ADDR_MOD_2` for load/store, while Blackhole uses `ADDR_MOD_7` / `ADDR_MOD_6`. These are different address modifier slot numbers but serve the same purpose (auto-incrementing the DST row pointer between iterations).

2. **Replay buffer API**: Blackhole uses `load_replay_buf(0, 10, [captures] { ... })` lambda syntax for recording the replay buffer, while Wormhole uses `lltt::record<lltt::NoExec>(0, 10)` followed by inline instructions. The functional behavior is identical.

3. **SFPSWAP stalling bug (Blackhole only)**: According to ISA documentation, Blackhole has a known hardware bug where `SFPSWAP` in all modes except `SFPSWAP_MOD1_SWAP` does not trigger automatic stalling when there is a data dependency with a preceding `SFPMAD`. Software must manually insert an `SFPNOP` to ensure a one-cycle gap. The SFPLOADMACRO scheduling in the init functions accounts for this.

4. **SFPLOADMACRO init functions**: Both architectures use identical SFPLOADMACRO template and macro configurations, including the same simple/mad/round/store bit patterns. The scheduling achieves the same 3-cycle (float) or 5-cycle (int32) throughput on both architectures.

---

## Runtime Arguments and Work Distribution

Runtime arguments are set by `set_eltwise_binary_runtime_args<true>()` (defined in `eltwise_multi_core_program_factory_common.hpp`). Key arguments per core:

- **Reader kernel**: `src0_addr`, `src1_addr`, `num_tiles`, `start_id`, `block_height`, `block_width`, `num_cores_y`
- **Writer kernel**: `output_addr`, `num_tiles`, `start_id`, `block_height`, `block_width`, `num_cores_y`
- **Compute kernel**: `per_core_block_cnt`, `per_core_block_size`

Work is distributed across cores in the `all_device_cores` grid (from `operation_attributes.worker_grid`). For interleaved tensors, tiles are split evenly with potential remainder on the last core. For sharded tensors, each core processes its local shard.

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: ElementWiseMultiCoreSfpu factory structure, kernel registration, compute config
- `tenstorrent/tt-llk`: `_llk_math_eltwise_binary_sfpu_params_` dispatch mechanism, SFPU init functions, VectorMode iteration
- `tenstorrent/tt-isa-documentation`: SFPSWAP instruction semantics, VEC_MIN_MAX mode, EXCHANGE_SRCB_SRCC bit, LaneConfig, Blackhole stalling bug
- `tenstorrent/sfpi`: `vec_min_max` function, `__builtin_rvtt_sfpswap` intrinsic, SFPSWAP_MOD1_VEC_MIN_MAX constant

### Confluence References
Not consulted for this analysis. DeepWiki provided sufficient SFPSWAP instruction detail.

### Glean References
Not consulted for this analysis. The open-source ckernel headers contained complete SFPU kernel implementations.

---

## File Reference Summary

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp` | Program factory: creates CBs, registers kernels, sets runtime args |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp` | Factory selection: `is_binary_sfpu_op()` and `select_program_factory()` |
| `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` | Define generation: `get_defines_fp32()` maps MAXIMUM to BINOP_INIT/BINARY_SFPU_OP |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp` | Runtime arg setup: `set_eltwise_binary_runtime_args()` |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | Reader kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel |
| `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` | Compute kernel |
| `tt_metal/hw/inc/api/compute/binary_max_min.h` | Compute API: `binary_max_tile()` and variants |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h` | LLK layer (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h` | LLK layer (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` | SFPU kernel (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` | SFPU kernel (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_init.h` | LLK init framework |
