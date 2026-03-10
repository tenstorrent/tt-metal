# ADD (element_wise_multi_core_sfpu) Implementation Analysis

## Overview

The ADD operation performs element-wise floating-point addition of two input tensors using the SFPU (Special Function Processing Unit) on Tenstorrent hardware. Unlike the FPU-based binary add (which uses the matrix engine's `add_tiles` instruction), this SFPU variant copies both input tiles into DEST registers and executes the addition via SFPI vector instructions. This program factory is selected when input tensor shapes are identical and the operation is identified as an SFPU-routed binary operation.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

One work unit is a **block of tiles**. The compute kernel processes tiles in blocks of size `per_core_block_size` (determined by `find_max_block_size`), iterating over `per_core_block_cnt` blocks per core. For the interleaved (non-sharded) case, `block_size = 1` and `block_cnt = num_tiles_per_core`, so one work unit equals one tile. For sharded tensors, the block size is the largest power-of-two divisor of `num_tiles_per_shard`.

## Tensor Format and Layout

### Input Tensors

| Property | Input A (cb_in0) | Input B (cb_in1) |
|---|---|---|
| Dimension Convention | NHWC-style, last dim is width | Same as A |
| Tensor Layout | TILED (32x32) | TILED (32x32) |
| Memory Layout | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | Any float (BF16, FP32, BF4_B, BF8_B) or INT32/UINT16/UINT32 | Same or different from A |

### Output Tensor

| Property | Output (cb_out0) |
|---|---|
| Dimension Convention | Same as inputs |
| Tensor Layout | TILED (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | Determined by operation config |

### Layout Transformations

No explicit tilize/untilize is performed within this program factory. All tensors are expected in tiled layout at entry. The `UnpackToDestMode::UnpackToDestFp32` is set for all CBs (except for POWER ops), meaning unpacking promotes data to FP32 in DEST registers regardless of the input data format.

## Data Flow Pattern

1. **Reader kernel** reads tiles from input tensor A into `cb_in0` (CB index 0) and from input tensor B into `cb_in1` (CB index 1). For sharded inputs, the reader simply does `cb_reserve_back` / `cb_push_back` to make the already-present L1 data available.
2. **Compute kernel** (for ADD with float types, no pre-scaling):
   - Waits for `per_core_block_size` tiles in both `cb_inp0` and `cb_inp1`.
   - Acquires DEST registers via `tile_regs_acquire()` + `tile_regs_wait()`.
   - Copies tiles from `cb_inp0` to even DEST slots (`i*2`) and from `cb_inp1` to odd DEST slots (`i*2+1`).
   - Calls `add_binary_tile_init()` then `add_binary_tile(i*2, i*2+1, i*2)` -- the SFPU adds DEST[i*2] + DEST[i*2+1] and writes result to DEST[i*2].
   - Packs result from DEST[i*2] into `cb_out0`.
   - Releases DEST registers and pops input CBs, pushes output CB.
3. **Writer kernel** reads tiles from `cb_out0` and writes them to the output tensor in DRAM/L1. For sharded output, it simply waits on the CB (data is already in the right L1 location).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Data Format | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | src0_cb_data_format | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (reader pushes 1 at a time) | Sharded: single-buffered (full shard); Interleaved: double-buffered | Reader | Compute |
| c_1 | cb_src1 | Input B tiles | src1_cb_data_format | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 | Same as c_0 | Reader | Compute |
| c_2 | cb_out0 | Output tiles | dst_cb_data_format | Sharded/block-width: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | `per_core_block_size` | Sharded: single-buffered; Interleaved: double-buffered | Compute | Writer |
| c_3 | cb_interm0 | Intermediate for pre-scaled input A | interim_cb0_format | `max_block_size` | `per_core_block_size` | Single-buffered | Compute (pre-scale) | Compute (main) |
| c_4 | cb_interm1 | Intermediate for pre-scaled input B | interim_cb1_format | `max_block_size` | `per_core_block_size` | Single-buffered | Compute (pre-scale) | Compute (main) |

**Note**: CBs c_3 and c_4 are only created when `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0` defines are present. For plain ADD, these defines are NOT set, so only c_0, c_1, and c_2 are used.

## Pipeline Pattern Summary

- **Interleaved path**: c_0 and c_1 have capacity `2 * max_block_size` with reader pushing 1 tile at a time -- this is **double-buffered**, allowing the reader to fill the next tile while compute processes the current one. c_2 has capacity `2 * max_block_size` -- also **double-buffered**.
- **Sharded path**: All CBs are sized to hold the entire shard, and the globally-allocated address maps directly to the tensor's L1 buffer. This is effectively **single-buffered** since the entire shard is available at once.

## Index Calculations

For **interleaved** tensors, the reader uses `TensorAccessor` to map linear tile IDs to physical DRAM/L1 addresses. Each core receives a `start_id` and `num_tiles`, iterating `tile_id` from `start_id` to `start_id + num_tiles - 1`. The `TensorAccessor` handles bank interleaving internally.

For **block/width-sharded** tensors, the `start_id` calculation accounts for the 2D shard grid:
```
start_id = (core_index / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_index % num_shards_per_width) * block_width
```
The reader then uses a nested loop over `block_height` x `block_width`, advancing the row start by `num_cores_y * block_width` after each row.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads via `noc_async_read_tile` with tile IDs incrementing from `start_id`. Each tile is read individually with a barrier after each read (no batching). This is a sequential access pattern.
- **Block/width sharded (reader for non-sharded input)**: Row-major within each block, with stride `num_cores_y * block_width` between rows.
- **Sharded inputs**: No NoC reads -- data is already in L1.

### Write Pattern
- **Interleaved**: Sequential tile writes via `noc_async_write_page` from `start_id` to `start_id + num_pages - 1`. Each tile is written individually with a flush after each write.
- **Block/width sharded to interleaved**: Row-major writes within unpadded block dimensions, skipping padding tiles.
- **Sharded output**: No NoC writes -- output CB is globally allocated to the output buffer.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Determined by `operation_attributes.worker_grid` |
| Work Splitting | `split_work_to_cores()` for interleaved; shard grid for sharded |
| Core Group 1 | Cores with `num_tiles_per_core_group_1` tiles (ceil division) |
| Core Group 2 | Cores with `num_tiles_per_core_group_2` tiles (floor division, may be 0) |
| Remainder Handling | Two core groups: group 1 gets one extra tile per core |
| Traversal Order | Row-major by default; shard orientation for sharded |
| Unused Cores | Runtime args zeroed out (num_tiles = 0) to create no-op execution |

For a single rectangular grid starting at (0,0), the factory uses an optimized path with `grid_to_cores()`. Otherwise, it falls back to `corerange_to_cores()` for arbitrary `CoreRangeSet` topologies.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if block/width sharded, 0 otherwise |
| 1+ | src0_args (TensorAccessorArgs) | varies | Tensor accessor params for input A (only if not IN0_SHARDED) |
| varies | src1_args (TensorAccessorArgs) | varies | Tensor accessor params for input B (only if not IN1_SHARDED) |

#### Writer Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (always c_2) |
| 1+ | dst_args (TensorAccessorArgs) | varies | Tensor accessor params for output buffer |

#### Compute Kernel
Compile-time args are passed via `ComputeConfig`:
- `fp32_dest_acc_en`: true if output is Float32, Int32, or UInt32
- `unpack_to_dest_mode`: `UnpackToDestFp32` for all CBs (for non-POWER ops)
- `defines`: Map of preprocessor defines including `BINOP_INIT`, `BINARY_SFPU_OP`, and optionally `PACK_RELU`

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Base address of input tensor A buffer |
| 1 | src1_addr | uint32_t | Base address of input tensor B buffer |
| 2 | num_tiles | uint32_t | Total tiles this core must read |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

#### Compute Kernel
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process on this core |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (interleaved output)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output tensor buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

#### Writer Kernel (block/width sharded to interleaved)
| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output tensor buffer |
| 1 | block_height_tiles | uint32_t | Block height in tiles |
| 2 | block_width_tiles | uint32_t | Block width in tiles |
| 3 | unpadded_block_height_tiles | uint32_t | Unpadded block height in tiles |
| 4 | unpadded_block_width_tiles | uint32_t | Unpadded block width in tiles |
| 5 | output_width_tiles | uint32_t | Full output width in tiles |
| 6 | block_num_tiles | uint32_t | Total tiles in the block |
| 7 | start_id_offset | uint32_t | Starting tile offset |
| 8 | start_id_base | uint32_t | Base starting tile ID (always 0) |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Handles 4 modes via preprocessor defines: (1) both sharded -- just reserve/push CBs, (2) IN0 sharded only, (3) IN1 sharded only, (4) neither sharded -- sequential tile reads. For block/width sharded non-sharded inputs, uses nested height x width loops with stride.

### Writer Kernel (interleaved)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page writes. For sharded output, just does `cb_wait_front` (no writes needed).

### Writer Kernel (block/width sharded to interleaved)

- **File**: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp`
- **Key Logic**: Waits for entire block, then writes only unpadded tiles in row-major order, skipping padding.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

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
#include "api/compute/eltwise_binary_sfpu.h"       // provides add_binary_tile, add_binary_tile_init, etc.
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

// PRE_SCALE is true if any pre-scaling defines are active (e.g., for LOGADDEXP, DIV, etc.)
// For plain ADD, neither SFPU_OP_INIT_PRE_IN0_0 nor SFPU_OP_INIT_PRE_IN1_0 is defined.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // number of blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // raw input A from reader
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // raw input B from reader

    // For ADD, no pre-scaling is needed, so cb_inp0 == cb_in0 and cb_inp1 == cb_in1.
    // When pre-scaling IS active (e.g., LOGADDEXP does exp on inputs first),
    // cb_inp0/cb_inp1 point to intermediate CBs c_3/c_4 that hold pre-scaled data.
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;
#else
    constexpr auto cb_inp0 = cb_in0;             // for ADD: cb_inp0 = c_0
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;
#else
    constexpr auto cb_inp1 = cb_in1;             // for ADD: cb_inp1 = c_1
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // output CB

    unary_op_init_common(cb_in0, cb_out0);        // initializes unpack/pack pipeline for these CBs

#ifdef PACK_RELU
    // For ADD + ReLU fused activation, configure pack stage to clamp negatives to zero.
    // This is a hardware-level optimization: ReLU is applied during packing, not as a separate SFPU op.
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // ---- PRE-SCALING SECTION (skipped for plain ADD) ----
        // For ops like LOGADDEXP, this section would apply exp() to each input
        // before the binary operation. For ADD, this entire section is compiled out.

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        cb_wait_front(cb_in0, per_core_block_size);
        cb_reserve_back(cb_inp0, per_core_block_size);

        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN0_0                       // e.g., exp_tile_init() for LOGADDEXP
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, i);                  // copy from cb_in0 to DEST[i]
            SFPU_OP_FUNC_PRE_IN0_0                    // e.g., exp_tile(i) for LOGADDEXP
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp0);                    // pack DEST[i] -> cb_inp0 (c_3)
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

        // ---- MAIN BINARY OPERATION SECTION ----

        cb_wait_front(cb_inp0, per_core_block_size);  // wait for input A tiles (from reader or pre-scale)
        cb_wait_front(cb_inp1, per_core_block_size);  // wait for input B tiles
        cb_reserve_back(cb_out0, per_core_block_size); // reserve space in output CB

        tile_regs_acquire();                           // acquire exclusive access to DEST registers
        tile_regs_wait();                              // wait until DEST registers are ready

        // Copy input A tiles into even DEST slots: DEST[0], DEST[2], DEST[4], ...
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0); // configure unpack for cb_inp0's data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);              // cb_inp0[i] -> DEST[i*2]
        }

        // Copy input B tiles into odd DEST slots: DEST[1], DEST[3], DEST[5], ...
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1); // reconfigure unpack for cb_inp1's data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);         // cb_inp1[i] -> DEST[i*2+1]

            // For ADD: BINOP_INIT expands to add_binary_tile_init()
            // This configures the SFPU for the binary add operation.
#ifdef BINOP_INIT
            BINOP_INIT                                 // add_binary_tile_init();
#endif
            // (Other init macros for INT ops, bitwise, shift are compiled out for float ADD)

            // For ADD: BINARY_SFPU_OP expands to add_binary_tile(i*2, i*2+1, i*2);
            // This performs DEST[i*2] = DEST[i*2] + DEST[i*2+1] on the SFPU.
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP                             // add_binary_tile(i*2, i*2+1, i*2);
#endif
            // Post-binary unary chain (e.g., for LOGADDEXP this would be log_tile)
            // For plain ADD, SFPU_OP_INIT_0 and SFPU_OP_CHAIN_0 are not defined.
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            pack_tile(i * 2, cb_out0);                 // pack DEST[i*2] (result) -> output CB
        }
        tile_regs_commit();                            // signal that DEST writes are complete
        tile_regs_release();                           // release DEST register lock

        cb_pop_front(cb_inp0, per_core_block_size);    // free input A tiles in CB
        cb_pop_front(cb_inp1, per_core_block_size);    // free input B tiles in CB
        cb_push_back(cb_out0, per_core_block_size);    // publish output tiles for writer
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to via `add_binary_tile(i*2, i*2+1, i*2)`.

#### SFPU Kernel File

`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h`

(Identical implementation exists for Blackhole at `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// (float32_to_bf16_rne and other binary variants omitted for brevity -- not used for ADD)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(
    const std::uint32_t dst_index_in0,    // DEST tile index for input A (i*2)
    const std::uint32_t dst_index_in1,    // DEST tile index for input B (i*2+1)
    const std::uint32_t dst_index_out)    // DEST tile index for output (i*2, same as in0)
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();

    // SFPU microcode: iterate over 8 rows per face (ITERATIONS=8).
    // Each tile has 4 faces, and the _llk_math_eltwise_binary_sfpu_params_ caller
    // invokes this function once per face (4 times in RC mode), advancing the
    // DEST register base pointer between faces via TTI_SETRWC.
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile occupies 32 "rows" in DEST when accessed via SFPI.
        // This is because DEST has 64 rows per tile but SFPI stride is 2,
        // so 64 / SFP_DESTREG_STRIDE(2) = 32.
        constexpr std::uint32_t dst_tile_size_sfpi = 32;

        // Load one row (32 floats) from DEST at the input A tile position.
        // dst_reg[dst_index_in0 * 32] indexes to the start of input A's tile.
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

        // Load one row (32 floats) from DEST at the input B tile position.
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = 0.0f;

        // Compile-time dispatch based on BINOP template parameter.
        // For BinaryOp::ADD, this is a simple vector addition.
        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1;   // SFPU vector add: 32-wide element-wise addition
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

        // Store the result back to DEST at the output tile position.
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the DEST register pointer to the next row within the face.
        // This increments the implicit row counter so the next iteration
        // processes the next row of 32 elements.
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    // For ADD, no special initialization is needed.
    // DIV and POW need reciprocal init; XLOGY needs log init.
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // BinaryOp::ADD falls through with no initialization.
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[index]` (load) | Loads a vector of 32 float elements from the DEST register file at the specified offset |
| `sfpi::dst_reg[index]` (store) | Stores a vector of 32 float elements back to the DEST register file |
| `sfpi::dst_reg++` | Increments the implicit DEST row pointer by 1 (advances to next row within a face) |
| `vFloat + vFloat` (operator+) | SFPU vector addition -- performs 32-wide element-wise floating-point addition |

For the ADD operation specifically, only the addition operator (`+`) on `vFloat` types is used. This maps to the SFPU's native floating-point addition capability operating on 32-element vectors.

#### SFPU Register Usage

- **DEST registers**: Two input tiles are loaded into adjacent DEST slots (e.g., DEST[0] and DEST[1]). The result overwrites the first input's slot (DEST[0]). Each tile occupies 32 SFPI-accessible rows in DEST.
- **dst_tile_size_sfpi = 32**: Constant offset multiplier to index different tiles within DEST. DEST has 64 physical rows per tile but SFPI accesses them with a stride of 2, yielding 32 addressable rows.
- **DEST row pointer** (`dst_reg++`): Auto-incremented each iteration to walk through the 8 rows of a face. The outer `_llk_math_eltwise_binary_sfpu_params_` function handles advancing between the 4 faces of a tile via `TTI_SETRWC`.

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` + `tile_regs_wait()` to get exclusive access to DEST registers.

2. **Unpack to DEST**: `copy_tile(cb_inp0, i, i*2)` unpacks tile `i` from `cb_inp0` into DEST slot `i*2`. Similarly, `copy_tile(cb_inp1, i, i*2+1)` unpacks tile `i` from `cb_inp1` into DEST slot `i*2+1`. With `UnpackToDestFp32` mode, data is promoted to FP32 in DEST regardless of input format.

3. **SFPU initialization**: `add_binary_tile_init()` calls through:
   - `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()` which calls
   - `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>(sfpu_binary_init<APPROX, BinaryOp::ADD>)` which calls
   - `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` -- configures SFPU config register, sets address modifiers (ADDR_MOD_7 with zero increments), resets counters
   - `sfpu_binary_init<APPROX, BinaryOp::ADD>()` -- calls `_sfpu_binary_init_` which is a no-op for ADD

4. **SFPU execution**: `add_binary_tile(i*2, i*2+1, i*2)` calls through:
   - `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>(i*2, i*2+1, i*2)` which calls
   - `_llk_math_eltwise_binary_sfpu_params_<APPROX>(calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, false>, i*2, i*2+1, i*2, VectorMode::RC)`
   - This function:
     a. Calls `_llk_math_eltwise_binary_sfpu_start_` -- sets DEST write address, stalls until math pipe ready
     b. Loops 4 times (once per face in RC mode), calling `_calculate_sfpu_binary_` each time
     c. After each face, advances the DEST pointer by 16 rows via two `TTI_SETRWC` commands (8 rows each)
     d. Calls `_llk_math_eltwise_binary_sfpu_done_` -- clears DEST address, waits for SFPU completion

5. **Inside `_calculate_sfpu_binary_`** (per face, 8 iterations):
   - Loads `in0` from DEST[dst_index_in0 * 32] (input A row)
   - Loads `in1` from DEST[dst_index_in1 * 32] (input B row)
   - Computes `result = in0 + in1` (32-wide vector add)
   - Stores `result` to DEST[dst_index_out * 32]
   - Increments `dst_reg` to next row

6. **Pack**: `pack_tile(i*2, cb_out0)` packs the result from DEST[i*2] back to the output circular buffer. The pack stage may apply ReLU clamping if `PACK_RELU` is defined.

#### SFPU Configuration

- **APPROXIMATION_MODE (APPROX)**: Template parameter passed through all layers. For ADD, it is unused since addition is exact.
- **fp32_dest_acc_en**: Set to `true` when output dtype is Float32/Int32/UInt32. This keeps DEST in full FP32 precision.
- **UnpackToDestFp32**: For all non-POWER binary SFPU ops, inputs are unpacked to FP32 in DEST, ensuring the SFPU operates in full precision.
- **ADDR_MOD_7**: Configured with zero increments for src/dest -- the SFPU manages its own addressing via `dst_reg` indexing.
- **VectorMode::RC**: Default mode processes all 4 faces of a 32x32 tile (row and column faces).

#### Hardware Compatibility Notes

The SFPU binary add implementation (`_calculate_sfpu_binary_` with `BinaryOp::ADD`) is identical between Wormhole B0 and Blackhole architectures. Both use the same SFPI intrinsics (`dst_reg` load/store, `vFloat` arithmetic). The LLK orchestration layer (`llk_math_eltwise_binary_sfpu.h` and `llk_math_eltwise_binary_sfpu_params.h`) is also architecture-identical for this operation.

The `ckernel_sfpu_binary.h` files in both `tt_llk_wormhole_b0` and `tt_llk_blackhole` contain the same implementation. Architecture differences would only manifest for operations requiring special hardware features (e.g., different LUT configurations or instruction availability), which do not apply to simple addition.

## Implementation Notes

1. **SFPU vs FPU path**: The SFPU binary add factory (`element_wise_multi_core_sfpu`) uses the SFPU vector unit rather than the FPU matrix unit. The key difference is that inputs are explicitly copied into DEST via `copy_tile` (using the A2D unpack path), and the SFPU then operates on DEST directly. The FPU path would use `add_tiles` which goes through the FPU's matrix pipeline. The SFPU path is used to support FP32 accumulation and mixed-precision operations that the FPU may not natively handle.

2. **Interleaved input pair layout in DEST**: Input A occupies even DEST slots (0, 2, 4, ...) and Input B occupies odd slots (1, 3, 5, ...). This interleaving ensures both operands of each tile pair are adjacent in DEST for efficient SFPU access. The result overwrites the even slot (Input A's position).

3. **DEST capacity constraint**: With 16-bit formats, up to 4 tiles from each operand can be in DEST simultaneously (8 total). With 32-bit formats, this is reduced to 2 tiles per operand (4 total). This limits `per_core_block_size` to at most 4 (16-bit) or 2 (32-bit) tiles per block.

4. **Fused ReLU optimization**: When the only fused activation is ReLU, it is applied at the pack stage via hardware (`PACK_RELU` define) rather than as a separate SFPU operation. This avoids an extra SFPU pass.

5. **Macro-driven polymorphism**: The same compute kernel source file handles ADD, SUB, MUL, DIV, POWER, bitwise, shift, comparison, and compound operations (LOGADDEXP, HYPOT, etc.) through preprocessor defines. For ADD, the active defines are:
   - `BINOP_INIT` = `add_binary_tile_init();`
   - `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2);`

6. **Integer ADD variant**: When both inputs are INT32, UINT32, or UINT16, the factory uses `add_int_tile` instead of `add_binary_tile`, with the `ADD_INT_INIT` define instead of `BINOP_INIT`. This routes through a different SFPU kernel optimized for integer arithmetic.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary eltwise SFPU program factory work in ttnn? What kernels does it use (reader, compute, writer)? How does it handle interleaved vs sharded tensors and core distribution for binary operations like add?"
   **Reason**: Initial architectural understanding of the factory structure and kernel selection.
   **Key Findings**: Confirmed the three kernels (reader_binary_interleaved_start_id, eltwise_binary_sfpu_kernel, writer_unary_interleaved_start_id), sharding handling via defines (IN0_SHARDED, IN1_SHARDED, OUT_SHARDED), and core distribution via worker_grid.

2. **Query**: "How does the SFPU binary add operation work at the ckernel level? What are the add_binary_tile_init and add_binary_tile functions?"
   **Reason**: Understanding the LLK-level implementation of the SFPU binary add.
   **Key Findings**: Confirmed the call chain: `add_binary_tile` -> `llk_math_eltwise_binary_sfpu_binop` -> `_llk_math_eltwise_binary_sfpu_params_` -> `_calculate_sfpu_binary_`. The ADD case simply does `result = in0 + in1`. Init is a no-op for ADD.

3. **Query**: "Where is the _calculate_sfpu_binary_ function defined? Show me the full implementation including BinaryOp::ADD handling."
   **Reason**: Locating the actual SFPU kernel source code in the tt-llk submodule.
   **Key Findings**: Found in `ckernel_sfpu_binary.h` within tt-llk. Confirmed the ITERATIONS=8 loop, dst_tile_size_sfpi=32, and the compile-time switch on BinaryOp.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp`
   **Reason**: Understanding how preprocessor defines are generated for the ADD operation.
   **Key Information**: For float ADD, `get_defines_fp32` sets `BINOP_INIT` = `add_binary_tile_init();` and `BINARY_SFPU_OP` = `add_binary_tile(i*2, i*2+1, i*2);`. For integer ADD, it uses `ADD_INT_INIT` with `add_int_tile` instead.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h`
   **Reason**: Understanding the LLK orchestration layer for binary SFPU operations.
   **Key Information**: `_llk_math_eltwise_binary_sfpu_init_` configures SFPU config register and address modifiers. `_llk_math_eltwise_binary_sfpu_start_` sets DEST write address and stalls until math ready. `_llk_math_eltwise_binary_sfpu_done_` clears state and waits for SFPU completion.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Understanding how the SFPU function is called across tile faces.
   **Key Information**: In RC mode, the SFPU function is called 4 times (once per face), with `TTI_SETRWC` advancing the DEST pointer by 16 rows between faces. This processes all 32x32 elements of a tile (4 faces x 8 iterations x 32-wide vectors = 1024 elements).
