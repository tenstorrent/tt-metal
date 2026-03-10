# DIV (element_wise_multi_core_sfpu) Implementation Analysis

## Overview

The DIV operation performs element-wise floating-point division of two tensors: `output = input_a / input_b`. It is implemented through the `BinaryDeviceOperation::ElementWiseMultiCoreSfpu` program factory, which routes the operation to the SFPU (vector engine) rather than the FPU (matrix unit). For floating-point types, the SFPU computes division as `in0 * reciprocal(in1)` with Newton-Raphson-refined reciprocal and special-case handling for division by zero, NaN, and equal-input cases. For INT32 types, a separate `div_int32_tile` path is used.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

One **work unit** is a single 32x32 tile. The total number of tiles is `a.physical_volume() / TILE_HW`. Tiles are grouped into blocks whose size is determined by `find_max_block_size()` for sharded tensors (largest power-of-2 dividing `num_tiles_per_shard`) or defaults to 1 for interleaved tensors. The compute kernel processes `per_core_block_cnt` blocks of `per_core_block_size` tiles each.

## Tensor Format and Layout

### Input Tensors

| Property | Input A (src0) | Input B (src1) |
|---|---|---|
| Dimension Convention | NHWC / row-major logical | NHWC / row-major logical |
| Tensor Layout | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32, INT32, etc. | BFLOAT16, FLOAT32, INT32, etc. |

If sharded: Shard Shape is `shard_spec.shape[0] x shard_spec.shape[1]`, Core Grid from `shard_spec.grid`, Shard Orientation is ROW_MAJOR or COL_MAJOR.

### Output Tensor

| Property | Output |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | Same as output dtype (BFLOAT16, FLOAT32, INT32, etc.) |

### Layout Transformations

No tilize/untilize or format conversions are performed by this program factory. All tensors must already be in tiled layout. However, when `is_fp32_dest_acc_en` is false (i.e., output is BFLOAT16), the SFPU kernel applies software Round-to-Nearest-Even (RNE) conversion from float32 to bfloat16 on the result before writing back to the DEST register.

## Data Flow Pattern

1. **Reader kernel** reads tiles from DRAM/L1 into CB_IN0 and CB_IN1 (one tile at a time for interleaved, or all tiles at once for sharded).
2. **Compute kernel** waits for tiles in CB_IN0 and CB_IN1, copies them to DEST registers (input A at even DEST slots `i*2`, input B at odd slots `i*2+1`), then calls `div_binary_tile(i*2, i*2+1, i*2)` which runs the SFPU division. Results are packed from DEST back to CB_OUT0.
3. **Writer kernel** reads tiles from CB_OUT0 and writes them to DRAM/L1 output buffer (one tile at a time for interleaved, or simply waits for sharded).

The DIV operation for floating-point types uses the `BINARY_SFPU_OP` path in the compute kernel. The `get_defines_fp32` function generates the define:
- `BINOP_INIT` = `div_binary_tile_init();`
- `BINARY_SFPU_OP` = `div_binary_tile(i*2, i*2+1, i*2);`

No pre-scaling stages (`SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`) are used for DIV in the fp32 path, so CB_INP0 = CB_IN0 and CB_INP1 = CB_IN1.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Data Format |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | sharded: `num_tiles_per_shard`; interleaved: `2 * max_block_size` | 1 (interleaved read) or `per_core_block_size` (compute) | Double-buffered (interleaved) or Single (sharded) | Reader | Compute | src0_cb_data_format |
| c_1 | cb_src1 | Input B tiles | sharded: `num_tiles_per_shard`; interleaved: `2 * max_block_size` | 1 (interleaved read) or `per_core_block_size` (compute) | Double-buffered (interleaved) or Single (sharded) | Reader | Compute | src1_cb_data_format |
| c_2 | cb_output | Output tiles | sharded/block_width_sharded: `num_tiles_per_shard`; interleaved: `2 * max_block_size` | `per_core_block_size` (compute) / 1 (writer) | Double-buffered (interleaved) or Single (sharded) | Compute | Writer | dst_cb_data_format |

Note: For the floating-point DIV path, intermediate CBs c_3 and c_4 are NOT created because no `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are generated.

## Pipeline Pattern Summary

For interleaved tensors with `max_block_size = 1`:
- CB_IN0 (c_0): capacity = 2 tiles, block = 1 tile -> **Double-buffered** (reader can write next tile while compute processes current)
- CB_IN1 (c_1): capacity = 2 tiles, block = 1 tile -> **Double-buffered**
- CB_OUT (c_2): capacity = 2 tiles, block = 1 tile -> **Double-buffered** (compute can produce next while writer drains current)

For sharded tensors: All tiles in the shard are present simultaneously, effectively single-buffered.

## Index Calculations

For **interleaved** (non-sharded) tensors, the reader iterates linearly from `start_id` to `start_id + num_tiles`, using `TensorAccessor` to map logical tile IDs to physical DRAM/L1 bank addresses.

For **block/width sharded** tensors, tiles are read in a 2D pattern: outer loop over `block_height` rows, inner loop over `block_width` columns. The tile ID advances by `num_cores_y * block_width` between rows (striding across shards distributed along the Y dimension).

The start tile ID for each core is computed as:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile NoC reads from DRAM banks. Each tile is read via `noc_async_read_tile` with a barrier after each tile.
- **Sharded**: No reads needed; tiles are already in L1 via the globally-allocated CB buffer. Reader simply does `cb_reserve_back` + `cb_push_back` to make tiles available.

### Write Pattern
- **Interleaved**: Sequential tile-by-tile NoC writes to DRAM banks via `noc_async_write_page`.
- **Sharded**: No writes needed; output CB is backed by the output tensor's L1 allocation. Writer does `cb_wait_front` on all tiles.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular grid from `operation_attributes.worker_grid` |
| Work Splitting | `tt::tt_metal::split_work_to_cores` for interleaved; shard grid for sharded |
| Core Group 1 | Cores handling `num_tiles_per_core_group_1` tiles each |
| Core Group 2 | Cores handling `num_tiles_per_core_group_2` tiles (remainder) |
| Load Balancing | Group 1 gets `ceil(total_tiles / num_cores)`, Group 2 gets `floor(total_tiles / num_cores)` |
| Remainder Handling | Extra tiles distributed to group 1 cores |
| Row Major | True for interleaved; from `shard_spec.orientation` for sharded |

An optimization path (`zero_start_grid`) is used when the worker grid is a single rectangular range starting at (0,0) and any shard grid also starts at (0,0), enabling faster work distribution algorithms.

## Arguments

### Compile-Time Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if input is block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs (src0) | uint32_t[] | Physical addressing info for input A (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs (src1) | uint32_t[] | Physical addressing info for input B (omitted if IN1_SHARDED) |

**Reader Kernel Defines:**

| Define | Condition | Value |
|---|---|---|
| IN0_SHARDED | input A is sharded | "1" |
| IN1_SHARDED | input B is sharded | "1" |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs (dst) | uint32_t[] | Physical addressing info for output buffer |

**Writer Kernel Defines:**

| Define | Condition | Value |
|---|---|---|
| OUT_SHARDED | output is sharded | "1" |

**Compute Kernel Defines (for floating-point DIV):**

| Define | Value | Description |
|---|---|---|
| BINOP_INIT | `div_binary_tile_init();` | Initializes SFPU for div operation |
| BINARY_SFPU_OP | `div_binary_tile(i*2, i*2+1, i*2);` | Performs SFPU division: DEST[i*2] = DEST[i*2] / DEST[i*2+1] |

**Compute Kernel Config:**

| Property | Value |
|---|---|
| fp32_dest_acc_en | True if output is Float32, Int32, or UInt32 |
| unpack_to_dest_mode | UnpackToDestFp32 for all CBs (non-POWER ops) |

### Runtime Arguments

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Input A buffer DRAM/L1 address |
| 1 | src1_addr | uint32_t | Input B buffer DRAM/L1 address |
| 2 | num_tiles | uint32_t | Total tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard height in tiles (0 if not block/width sharded) |
| 5 | block_width | uint32_t | Shard width in tiles (0 if not block/width sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks for this core |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

**Writer Kernel (interleaved output):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Output buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Total tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Handles four variants via compile-time defines:
  - Both inputs interleaved: reads tiles sequentially from DRAM for both inputs
  - IN0 sharded: skips reading input A (already in L1), reads only input B
  - IN1 sharded: reads only input A, skips input B
  - Both sharded: no DRAM reads, just makes sharded data available via CB reserve/push
  - Block/width sharded path uses 2D tile addressing (row x column with stride)

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential tile writer. For sharded output, just waits for all tiles. For interleaved, iterates and writes one tile at a time via NoC.

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
#include "api/compute/eltwise_binary_sfpu.h"    // provides div_binary_tile, div_binary_tile_init
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

// PRE_SCALE is true if any pre-scaling stage is defined (not used for fp DIV)
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // number of blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // CB for input A
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // CB for input B

    // For DIV (fp32 path): no PRE defines, so cb_inp0 = cb_in0, cb_inp1 = cb_in1
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;   // interim CB for pre-processed input A
#else
    constexpr auto cb_inp0 = cb_in0;              // direct: no pre-processing for input A
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;   // interim CB for pre-processed input B
#else
    constexpr auto cb_inp1 = cb_in1;              // direct: no pre-processing for input B
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // output CB

    unary_op_init_common(cb_in0, cb_out0);        // initializes unpacker and packer for these CBs

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // fused ReLU on pack (not used for plain DIV)
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- PRE_SCALE stages would go here if defined (skipped for fp DIV) ---

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

        // For DIV, pre-scaling stages are not compiled, so we skip directly to the main loop.

        // --- MAIN BINARY SFPU COMPUTATION ---
        cb_wait_front(cb_inp0, per_core_block_size);   // wait for input A tiles from reader
        cb_wait_front(cb_inp1, per_core_block_size);   // wait for input B tiles from reader
        cb_reserve_back(cb_out0, per_core_block_size);  // reserve output space

        tile_regs_acquire();   // acquire DEST register file for writing
        tile_regs_wait();      // wait for DEST registers to be available

        // Copy input A tiles into DEST at even indices (0, 2, 4, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);  // configure unpacker for cb_inp0 format
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);   // CB_IN0[i] -> DEST[i*2]
        }

        // Copy input B tiles into DEST at odd indices (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);  // reconfigure unpacker for cb_inp1 format
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);   // CB_IN1[i] -> DEST[i*2+1]

            // For DIV: BINOP_INIT expands to div_binary_tile_init() -- initializes SFPU reciprocal
            // BINARY_SFPU_OP expands to div_binary_tile(i*2, i*2+1, i*2)
            //   which computes DEST[i*2] = DEST[i*2] / DEST[i*2+1] via SFPU
            BINOP_INIT                           // div_binary_tile_init();
            BINARY_SFPU_OP                       // div_binary_tile(i*2, i*2+1, i*2);

            // No SFPU_OP_INIT_0 / SFPU_OP_FUNC_0 / SFPU_OP_CHAIN_0 for plain DIV

            pack_tile(i * 2, cb_out0);           // pack result from DEST[i*2] -> CB_OUT0
        }
        tile_regs_commit();    // signal DEST writes complete
        tile_regs_release();   // release DEST register file

        cb_pop_front(cb_inp0, per_core_block_size);    // free input A tiles in CB
        cb_pop_front(cb_inp1, per_core_block_size);    // free input B tiles in CB
        cb_push_back(cb_out0, per_core_block_size);    // publish output tiles for writer
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`

(Wormhole B0 version is at `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` and is identical.)

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
// Implements the "add 0x7fff + LSB" algorithm for correct tie-breaking.
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);     // reinterpret float as uint32
    sfpi::vUInt lsb = (bits >> 16) & 1;                         // extract bit 16 (bf16 mantissa LSB)
    bits = bits + 0x7fffU + lsb;                                 // RNE rounding: add bias + LSB for tie-break
    bits = bits & 0xFFFF0000U;                                   // zero lower 16 bits to get bf16 in upper half
    return sfpi::reinterpret<sfpi::vFloat>(bits);                // reinterpret back as float
}

// Generic binary SFPU dispatch (ADD, SUB, DIV, RSUB, etc.)
// For DIV: this calls _calculate_sfpu_binary_ which is defined in tt_llk.
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
}

// Specialized MUL implementation (not used for DIV, shown for context)
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;   // 64 / SFP_DESTREG_STRIDE = 32 rows per tile face
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 * in1;
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);          // truncate to bf16 precision
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; } // exact zero handling to match FPU
            v_endif;
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;   // advance to next row within the tile face
    }
}

// DIV-specific SFPU implementation: result = in0 * reciprocal(in1)
// This is the function called by div_binary_tile -> llk_math_eltwise_binary_sfpu_binop_div
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;   // each tile in DEST spans 32 SFPI rows
    for (int d = 0; d < ITERATIONS; d++) {    // iterate over 8 row-groups per tile face
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];  // load element from input A
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];  // load element from input B

        // Core division: multiply in0 by 2-iteration Newton-Raphson reciprocal of in1
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1);

        // Special case: division by zero
        v_if(in1 == 0) {
            v_if(in0 == 0) {
                result = std::numeric_limits<float>::quiet_NaN();  // 0/0 = NaN
            }
            v_else {
                result = std::numeric_limits<float>::infinity();   // x/0 = +/-inf
                result = sfpi::setsgn(result, in0);                // sign matches numerator
            }
            v_endif;
        }
        v_elseif(in0 == in1) {
            result = sfpi::vConst1;   // x/x = 1.0 (handles precision edge case)
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);   // RNE conversion to bf16 when not in fp32 dest mode
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;  // write result back to DEST
        sfpi::dst_reg++;   // advance SFPU register pointer to next row
    }
}

// Binary SFPU init: for DIV, calls _init_sfpu_reciprocal_<false>()
template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
    // For BinaryOp::DIV, this expands to _init_sfpu_reciprocal_<false>()
    // which loads vConstFloatPrgm0 = 2.0f (Blackhole) or polynomial coefficients (Wormhole)
    // needed by the Newton-Raphson reciprocal iterations.
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[index]` (load) | Loads a vector float element from DEST register at the given offset |
| `sfpi::dst_reg[index]` (store) | Stores a vector float result back to DEST register |
| `sfpi::dst_reg++` | Advances the SFPU register pointer to the next row within the tile face |
| `_sfpu_reciprocal_<2>(in1)` | Computes 1/in1 using initial approximation + 2 Newton-Raphson iterations |
| `sfpi::vFloat * sfpi::vFloat` | SFPU floating-point vector multiply |
| `sfpi::setsgn(result, in0)` | Copies the sign bit from `in0` to `result` |
| `sfpi::reinterpret<vUInt>(vFloat)` | Reinterprets float vector as unsigned int vector (bitcast) |
| `sfpi::reinterpret<vFloat>(vUInt)` | Reinterprets unsigned int vector as float vector (bitcast) |
| `v_if / v_else / v_elseif / v_endif` | SFPU conditional execution (predicated lanes) |
| `sfpi::vConst1` | SFPU constant 1.0f |
| `sfpi::approx_recip(x)` | Initial reciprocal approximation (hardware LUT, used inside `_sfpu_reciprocal_`) |

#### SFPU Register Usage

- **DEST registers**: Tiles are loaded into DEST with input A at even indices (0, 2, 4, ...) and input B at odd indices (1, 3, 5, ...). The output overwrites input A's DEST slot (even index).
- **dst_tile_size_sfpi = 32**: Each tile occupies 32 SFPI-addressable rows in DEST (64 rows / SFP_DESTREG_STRIDE of 2).
- **ITERATIONS = 8**: The SFPU loop iterates 8 times per tile face (32 rows / 4 elements per SFPU vector = 8 iterations), processing 4 elements per iteration via SIMD vector operations.
- **vConstFloatPrgm0**: Set to 2.0f during init (Blackhole), used by Newton-Raphson reciprocal refinement.
- **vConstFloatPrgm0/1/2**: Set to polynomial coefficients during init (Wormhole B0), used for quadratic initial reciprocal estimate.

#### SFPU Execution Flow

1. **Initialization**: `div_binary_tile_init()` -> `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::DIV>()` -> `_sfpu_binary_init_<APPROX, DIV>()` -> `_init_sfpu_reciprocal_<false>()`. This loads SFPU programmable constants needed for reciprocal computation.

2. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` and `tile_regs_wait()` to gain exclusive access to the DEST register file.

3. **Unpack to DEST**: `copy_tile(cb_inp0, i, i*2)` unpacks input A tile `i` from CB into DEST slot `i*2`. `copy_tile(cb_inp1, i, i*2+1)` unpacks input B tile `i` into DEST slot `i*2+1`. The `UnpackToDestFp32` mode ensures data is promoted to fp32 in DEST.

4. **SFPU math**: `div_binary_tile(i*2, i*2+1, i*2)` invokes the LLK layer:
   - `llk_math_eltwise_binary_sfpu_binop_div<APPROX, DIV, DST_ACCUM_MODE>(i*2, i*2+1, i*2)`
   - This calls `_llk_math_eltwise_binary_sfpu_params_` which iterates over all 4 faces of the tile (each face has 8 SFPU iterations for 32 rows).
   - For each iteration: loads `in0` and `in1` from DEST, computes `in0 * reciprocal(in1)`, handles special cases (0/0->NaN, x/0->inf, x/x->1), optionally converts to bf16, writes result back.

5. **Pack**: `pack_tile(i*2, cb_out0)` reads the result from DEST slot `i*2` and packs it into the output circular buffer.

6. **Release**: `tile_regs_commit()` and `tile_regs_release()` free the DEST register file. `cb_pop_front` frees input CBs, `cb_push_back` publishes output to writer.

#### SFPU Configuration

| Configuration | Value | Description |
|---|---|---|
| `fp32_dest_acc_en` | True for Float32/Int32/UInt32 output | Keeps DEST in fp32 mode, skips bf16 RNE conversion |
| `unpack_to_dest_mode` | `UnpackToDestFp32` for all input CBs | Promotes input data to fp32 in DEST registers |
| `APPROX` | Compile-time constant from ComputeConfig | Controls approximation mode for reciprocal (typically false) |
| `_init_sfpu_reciprocal_<false>()` | Loads NR constants | BH: vConstFloatPrgm0=2.0f; WH: loads quadratic coefficients |
| `_sfpu_reciprocal_<2>()` | 2 Newton-Raphson iterations | Provides high-accuracy reciprocal for division |

#### Hardware Compatibility Notes

- **Blackhole**: `_sfpu_reciprocal_<2>` uses `sfpi::approx_recip()` as initial estimate, then performs 2 Newton-Raphson iterations using `t = x * y - vConstFloatPrgm0` and `y = y * -t - vConst0`.
- **Wormhole B0**: `_sfpu_reciprocal_<2>` uses a quadratic polynomial initial estimate (3 programmable constants), then performs 2 Newton-Raphson iterations using `t = 1 + negative_x * y` and `y = y + y * t`, followed by scale and sign correction.
- Both architectures produce the same functional result but via different microcode sequences tuned to their respective SFPU hardware.
- The `ckernel_sfpu_binary.h` file is identical between Blackhole and Wormhole B0 -- the architectural differences are encapsulated in the `_sfpu_reciprocal_` implementation (in the tt_llk submodule).

## Implementation Notes

1. **DIV is computed as multiply-by-reciprocal**: The SFPU does not have a native division instruction. Instead, it computes `1/in1` via Newton-Raphson refinement of a hardware reciprocal approximation, then multiplies by `in0`.

2. **Special case handling**: The kernel explicitly handles three edge cases:
   - `0/0` -> NaN
   - `x/0` (x != 0) -> +/-infinity with sign matching `x`
   - `x/x` -> exactly 1.0 (avoids floating-point precision artifacts from the reciprocal approximation)

3. **bf16 RNE conversion**: When `is_fp32_dest_acc_en` is false (BFLOAT16 output), the result undergoes software Round-to-Nearest-Even conversion. This is more accurate than simple truncation and matches IEEE 754 rounding behavior.

4. **Per-tile init/op pattern**: Note that `BINOP_INIT` (div_binary_tile_init) is called inside the inner loop for each tile. This is because the init function sets SFPU programmable constants, and other operations in the pipeline might overwrite them.

5. **UnpackToDestFp32**: For non-POWER binary SFPU ops, all inputs are promoted to fp32 in DEST regardless of input format. This ensures the SFPU operates at full fp32 precision internally.

6. **Dual program factory paths**: The `get_defines_fp32` function (used by this SFPU program factory) generates a direct `div_binary_tile` call. The older `get_defines` function (used by the FPU program factory) instead generates a RECIP pre-scale on input B followed by FPU multiply, which is a fundamentally different execution strategy.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary element-wise SFPU program factory work in ttnn? What kernels does it use and how does it distribute work across cores?"
   **Reason**: Needed to understand the overall program factory architecture and kernel selection.
   **Key Findings**: Confirmed three-kernel architecture (reader, compute, writer), SPMD work distribution via `split_work_to_cores`, and the specific kernel file paths.

2. **Query**: "What is the SFPU div operation implementation in tt-metal? How does binary SFPU div work at the kernel level?"
   **Reason**: Needed to understand the division implementation at the SFPU level.
   **Key Findings**: Division uses `in0 * _sfpu_reciprocal_<2>(in1)` with special-case handling for 0/0, x/0, and x/x. Located the ckernel_sfpu_binary.h files.

3. **Query**: "What is the implementation of _sfpu_reciprocal_ template function?"
   **Reason**: Needed to understand the reciprocal computation that underlies division.
   **Key Findings**: Blackhole uses `approx_recip()` + Newton-Raphson; Wormhole B0 uses quadratic polynomial estimate + Newton-Raphson. The `max_iter` parameter (set to 2 for DIV) controls precision.

4. **Query**: "What does _sfpu_binary_init_ do for BinaryOp::DIV?"
   **Reason**: Needed to understand initialization of the SFPU for division.
   **Key Findings**: For DIV, it calls `_init_sfpu_reciprocal_<false>()` which loads the programmable constants needed by the Newton-Raphson reciprocal iterations.

5. **Query**: "What is _llk_math_eltwise_binary_sfpu_params_ and how does it iterate over tile faces?"
   **Reason**: Needed to understand the LLK layer that orchestrates SFPU execution across tile faces.
   **Key Findings**: It passes the SFPU calculation function and destination indices to iterate across all 4 faces of a tile (8 SFPU iterations per face, 32 rows total).

### Confluence References

Not consulted for this analysis. The DeepWiki and source code provided sufficient detail for all SFPU instruction semantics.

### Glean References

Not consulted for this analysis.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Needed to understand how DIV maps to SFPU defines.
   **Key Information**: `get_defines_fp32` for `BinaryOpType::DIV` generates `BINOP_INIT = "div_binary_tile_init();"` and `BINARY_SFPU_OP = "div_binary_tile(i*2, i*2+1, i*2);"`. For INT32, it uses `div_int32_tile` instead.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Needed to trace the API call chain from `div_binary_tile` to LLK layer.
   **Key Information**: `div_binary_tile()` calls `llk_math_eltwise_binary_sfpu_binop_div<APPROX, BinaryOp::DIV, DST_ACCUM_MODE>()`.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h`
   **Reason**: Needed to understand the LLK dispatch for div.
   **Key Information**: `llk_math_eltwise_binary_sfpu_binop_div` passes `calculate_sfpu_binary_div` to `_llk_math_eltwise_binary_sfpu_params_` for tile-face iteration.

4. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Needed to understand runtime argument setup and work distribution logic.
   **Key Information**: Detailed core distribution with two groups, sharded vs interleaved paths, and block/width shard addressing.
