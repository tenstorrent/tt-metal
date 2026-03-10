# WHERE Operation Implementation Analysis

## Overview

The WHERE operation implements an element-wise conditional selection: `output[i] = predicate[i] ? value_true[i] : value_false[i]`. It is a ternary SFPU operation that evaluates a condition tensor and selects between two values (tensors or scalars) on a per-element basis.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

The implementation supports three input variants:
- **TTT** (Tensor-Tensor-Tensor): All three inputs are tensors
- **TTS** (Tensor-Tensor-Scalar): Predicate and true-value are tensors, false-value is a scalar
- **TST** (Tensor-Scalar-Tensor): Predicate and false-value are tensors, true-value is a scalar

Additionally, each variant supports multiple broadcast types: NONE, OUTER_BCAST, COL_BCAST, ROW_BCAST, SCALAR_BCAST, SCALAR_A_BCAST, and SCALAR_B_BCAST.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `output.physical_volume() / tile_hw` |
| **Loop structure** | For each output tile: read condition+true+false tiles, compute where, write output tile |

One work unit is a single 32x32 tile. The compute kernel processes one tile per cycle: it loads three tiles (predicate, true-value, false-value) into destination registers, applies the SFPU where operation, and packs the result to the output circular buffer.

## Tensor Format and Layout

### Input Tensors

| Property | Predicate Tensor (input_a) | True-Value Tensor (input_b) | False-Value Tensor (input_c) |
|----------|---------------------------|----------------------------|------------------------------|
| **Logical shape** | Up to 5D+ (collapsed for rank > 5) | Same or broadcastable | Same or broadcastable |
| **Dimension convention** | [..., D, N, C, H, W] | [..., D, N, C, H, W] | [..., D, N, C, H, W] |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, UINT32 | BFLOAT16, FLOAT32, INT32, UINT32 | BFLOAT16, FLOAT32, INT32, UINT32 |

Notes:
- For TTS variant, input_c is a scalar (float, int32_t, or uint32_t) rather than a tensor
- For TST variant, input_b is a scalar
- Inputs can be independently sharded or interleaved
- When sharded, native L1 sharding requires: all tensors share the same shape/memory config, same grid, even shard sizes, and L1 buffer type

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Broadcast-expanded shape of all inputs |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Matches input dtype or specified output_dtype |

### Layout Transformations

No tilize/untilize conversions are performed. All inputs and outputs must already be in TILE_LAYOUT. For broadcast cases, the reader kernels handle tile-level broadcasting by repeating reads of single-dimension tiles (column broadcast fills tile from first column, row broadcast from first row, scalar broadcast from first element). For the ROW_BCAST+BF16 special case, the LLK `unary_bcast<BroadcastType::ROW>` is used in the compute kernel to expand row-broadcast tiles.

## Data Flow Pattern

### TTT No-Broadcast (simplest case)

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (predicate) | c_0 | reserve_back(c_0,1), noc_async_read, push_back(c_0,1) |
| 1 | Reader | DRAM/L1 (true) | c_1 | reserve_back(c_1,1), noc_async_read, push_back(c_1,1) |
| 1 | Reader | DRAM/L1 (false) | c_2 | reserve_back(c_2,1), noc_async_read, push_back(c_2,1) |
| 2 | Compute | c_0, c_1, c_2 | c_3 | wait_front(c_0/1/2,1), copy_tile to DST[0,1,2], where_tile(0,1,2,0), pack_tile(0,c_3), push_back(c_3,1), pop_front(c_0/1/2,1) |
| 3 | Writer | c_3 | DRAM/L1 (output) | wait_front(c_3,1), noc_async_write, pop_front(c_3,1) |

### TTS/TST No-Broadcast

Same as TTT but only two tensor CBs (c_0 for predicate, c_1 for the tensor operand). The scalar value is filled into the appropriate destination register using `fill_tile` in the compute kernel.

### Broadcast Variants (COL_BCAST, SCALAR_BCAST)

For column and scalar broadcast, the compute kernel uses a `process_tile` function with an outer loop structure. Broadcast tensors are waited on once outside the loop and popped once after the loop completes, while non-broadcast tensors are waited/popped per iteration.

### ROW_BCAST (BF16-only LLK path)

When all three inputs are BF16 and ROW_BCAST is detected, the compute kernel uses intermediate CBs (c_4, c_5, c_6) and `unary_bcast<BroadcastType::ROW>` to expand row-broadcast inputs before the ternary SFPU operation.

## Circular Buffer Configuration

### TTT Variant (No Broadcast / Outer Broadcast)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | predicate_tensor_cb | Predicate input staging | 2 tiles (or shard volume) | 1 tile | Double | Reader | Compute | Block |
| c_1 | value_true_tensor_cb | True-value input staging | 2 tiles (or shard volume) | 1 tile | Double | Reader | Compute | Block |
| c_2 | value_false_tensor_cb | False-value input staging | 2 tiles (or shard volume) | 1 tile | Double | Reader | Compute | Block |
| c_3 | output_tensor_cb | Output staging | 2 tiles (or shard volume) | 1 tile | Double | Compute | Writer | Block |

### TTT ROW_BCAST (BF16 only, additional CBs)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_4 | cb_bcast_a | Row-broadcast expanded predicate | 2 tiles | 1 tile | Double | Compute (bcast stage) | Compute (SFPU stage) | Block |
| c_5 | cb_bcast_b | Row-broadcast expanded true-value | 2 tiles | 1 tile | Double | Compute (bcast stage) | Compute (SFPU stage) | Block |
| c_6 | cb_bcast_c | Row-broadcast expanded false-value | 2 tiles | 1 tile | Double | Compute (bcast stage) | Compute (SFPU stage) | Block |

### TTS/TST Variant

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | predicate_tensor_cb | Predicate input staging | 2 tiles (or shard volume) | 1 tile | Double | Reader | Compute | Block |
| c_1 | tensor_cb | Tensor operand staging (true for TTS, false for TST) | 2 tiles (or shard volume) | 1 tile | Double | Reader | Compute | Block |
| c_3 | output_tensor_cb | Output staging | 2 tiles (or shard volume) | 1 tile | Double | Compute | Writer | Block |

When sharded, the CB capacity equals the shard volume in tiles rather than 2 tiles, and the CB is backed by the tensor's L1 buffer directly.

## Pipeline Pattern Summary

- **Interleaved mode**: All CBs are double-buffered (capacity=2, block=1), enabling the reader to fill the next tile while compute processes the current one.
- **Sharded mode**: CBs are sized to the full shard volume. The reader pushes all shard tiles at once (for sharded inputs) and the compute processes them sequentially.
- The reader, compute, and writer operate in a pipelined fashion with single-tile granularity for interleaved mode.

## Index Calculations

The program factory uses a multi-dimensional nested loop (ND, D, N, C, H, W) for tile index calculation, supporting tensors up to rank 5+ (rank > 5 dimensions are collapsed into a single ND dimension).

For each input tensor, strides are computed as:
- `nD_stride = Ht * Wt * C * N * D * (ND > 1)`
- `d_stride = Ht * Wt * C * N * (D > 1)`
- `n_stride = Ht * Wt * C * (N > 1)`
- `c_stride = Ht * Wt * (C > 1)`

A stride of 0 when the dimension is 1 enables outer-dimension broadcasting: the tile offset does not advance for broadcast dimensions.

For width-sharded mode, the writer and reader use `dst_shard_width` to limit tile iteration within each row, with `dst_tile_offset` adjustments to account for skipped tiles.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads via `noc_async_read_page` with a read barrier after each tile (or batch of 3 tiles for TTT). Tiles are read in row-major order within the tile grid (W dimension innermost).
- **Sharded**: All shard tiles are pushed to the CB at once via `cb_reserve_back` / `cb_push_back` with no NoC reads (data is already in L1).
- **Broadcast**: For column/scalar broadcast, the reader kernel fills tiles by replicating the first column/element using dataflow utility functions (e.g., `fill_tile_with_first_column`, `fill_tile_with_first_element`).

### Write Pattern
- **Interleaved**: Sequential tile-by-tile writes via `noc_async_write_page` with a write barrier per tile.
- **Sharded**: Output CB is directly backed by the output tensor's L1 buffer. No explicit writes needed (DST_SHARDED define skips the write loop entirely).

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (preferred) or arbitrary CoreRangeSet |
| **Grid dimensions** | Determined by `operation_attributes.worker_grid` |
| **Total cores** | `worker_grid.num_cores()` or `compute_with_storage_grid.x * compute_with_storage_grid.y` |
| **Work per core** | `num_tiles_per_core_group_1` or `num_tiles_per_core_group_2` (interleaved); shard volume (sharded) |
| **Load balancing** | Two-group split: group_1 gets `ceil(total_tiles / num_cores)`, group_2 gets `floor(total_tiles / num_cores)` |

**Interleaved mode**: Uses `split_work_to_cores` to divide output tiles across cores. Creates two core groups -- group_1 with one extra tile per core if tiles don't divide evenly. Cores not in either group receive zero-initialized runtime args and produce no output.

**Sharded mode**: Core grid is determined by the shard spec. Each core processes its local shard. The `ShardShapeGenerator` class handles uneven shards at grid edges (last row/column of cores may have fewer tiles). A `zero_start_grid` optimization is used when the grid starts at (0,0) for more efficient core enumeration.

## Arguments

### Compile-Time Arguments

#### Reader Kernel (TTT)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src0 | uint32_t | CB index for predicate tensor (c_0) |
| 1 | cb_id_src1 | uint32_t | CB index for true-value tensor (c_1) |
| 2 | cb_id_src2 | uint32_t | CB index for false-value tensor (c_2) |
| 3+ | TensorAccessorArgs (src0) | varies | Compile-time tensor accessor parameters for predicate |
| ... | TensorAccessorArgs (src1) | varies | Compile-time tensor accessor parameters for true-value |
| ... | TensorAccessorArgs (src2) | varies | Compile-time tensor accessor parameters for false-value |
| last | has_sharding | uint32_t | 1 if sharding is active, 0 otherwise |

#### Reader Kernel (TTS/TST)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_src0 | uint32_t | CB index for predicate tensor (c_0) |
| 1 | cb_id_src1 | uint32_t | CB index for tensor operand (c_1) |
| 2+ | TensorAccessorArgs (src0) | varies | Compile-time tensor accessor parameters for predicate |
| ... | TensorAccessorArgs (src1) | varies | Compile-time tensor accessor parameters for tensor operand |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | CB index for output tensor (c_3) |
| 1+ | TensorAccessorArgs (dst) | varies | Compile-time tensor accessor parameters for output |
| last | has_sharding | uint32_t | 1 if sharding is active, 0 otherwise |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- tiles processed per read-compute-write cycle |
| 1 | scalar_is_true_value | uint32_t | 1 for TST (scalar = true value), 0 for TTS (scalar = false value). Only used by TTS/TST kernels. |

### Runtime Arguments

#### Reader (27 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | DRAM/L1 address of predicate tensor buffer |
| 1 | src1_addr | uint32_t | DRAM/L1 address of true-value tensor buffer (or tensor operand for TTS/TST) |
| 2 | src2_addr | uint32_t | DRAM/L1 address of false-value tensor buffer (0 for TTS/TST) |
| 3 | num_tiles | uint32_t | Number of tiles to process on this core |
| 4 | start_id | uint32_t | Starting output tile index for this core |
| 5-8 | pred strides | uint32_t | nD_stride, d_stride, n_stride, c_stride for predicate |
| 9-14 | output dims | uint32_t | D, N, C, Ht, Wt, cND of output tensor |
| 15-19 | true/tensor strides | uint32_t | nD_stride, d_stride, n_stride, c_stride, num_tiles for true tensor (or tensor operand) |
| 20-24 | false/scalar strides | uint32_t | nD_stride, d_stride, n_stride, c_stride, num_tiles for false tensor (0 for TTS/TST) |
| 25 | dst_shard_width | uint32_t | Output shard width in tiles (0 if not sharded) |
| 26 | src_num_tiles | uint32_t | Predicate shard tile count (0 if not sharded) |

#### Writer (11 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output tensor buffer |
| 1 | num_tiles | uint32_t | Number of tiles to write on this core |
| 2 | start_id | uint32_t | Starting output tile index |
| 3 | dst_shard_width | uint32_t | Output shard width in tiles |
| 4-9 | output dims | uint32_t | D, N, C, Ht, Wt, cND |
| 10 | padding | uint32_t | Reserved (0) |

#### Compute (4 args per core)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | num_tiles | uint32_t | Number of tiles to process on this core |
| 1 | freq | uint32_t | Broadcast frequency (tiles before re-reading broadcast source). 0 for no-broadcast. |
| 2 | counter | uint32_t | Starting offset within the broadcast frequency window |
| 3 | scalar_arg | uint32_t | Packed scalar value (bit-cast float/int32/uint32). 0 for TTT variant. |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| Reader | RISCV_0 | NOC0 | DRAM/L1 tensors | c_0, c_1, [c_2] | Read predicate/true/false tiles via NoC, or push sharded tiles |
| Compute | RISCV_2 | N/A | c_0, c_1, [c_2] | c_3 | copy_tile to DEST, where_tile SFPU op, pack_tile |
| Writer | RISCV_1 | NOC1 | c_3 | DRAM/L1 output | Write output tiles via NoC, or no-op for sharded output |

### Reader Kernel (TTT No-Broadcast)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp`
- **Key Logic**: Uses TensorAccessor for address calculation. Iterates through ND,D,N,C,H,W dimensions with stride-based tile offset computation. For sharded inputs, simply does `cb_reserve_back`/`cb_push_back` to make L1 data available. For non-sharded inputs, reads tile-by-tile via `noc_async_read_page`. Width sharding is supported by limiting the W-tile loop to `dst_shard_width`.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp`
- **Key Logic**: When `DST_SHARDED` is defined, the entire kernel body is skipped (output is already in L1). Otherwise, iterates through ND,D,N,C,H,W dimensions writing tiles via `noc_async_write_page` with per-tile write barrier.

### Compute Kernel

This section focuses on the primary TTT no-broadcast compute kernel. Other compute kernel variants (broadcast, TTS/TST) follow the same SFPU pattern but with different CB synchronization logic.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"  // Provides where_tile() and where_tile_init()
#include "api/compute/eltwise_unary/lerp.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);  // runtime arg 0: total tiles for this core

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // always 1: one tile per iteration

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;  // predicate tensor CB
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;  // true-value tensor CB
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;  // false-value tensor CB
    constexpr auto cb_out = tt::CBIndex::c_3;       // output CB

    // Initialize unpack/pack hardware for SFPU operation path
    unary_op_init_common(cb_pre_in1, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for reader to produce one tile in each input CB
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);  // block until predicate tile available
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);  // block until true-value tile available
        cb_wait_front(cb_pre_in3, num_tiles_per_cycle);  // block until false-value tile available

        // Reserve space in output CB for one tile
        cb_reserve_back(cb_out, num_tiles_per_cycle);

        // Acquire exclusive access to destination registers (DEST)
        tile_regs_acquire();

        // Unpack predicate tile from c_0 into DEST register 0
        copy_tile_to_dst_init_short(cb_pre_in1);  // configure unpack for c_0's data format
        copy_tile(cb_pre_in1, 0, 0);              // copy tile 0 from c_0 to DEST[0]

        // Unpack true-value tile from c_1 into DEST register 1
        copy_tile_to_dst_init_short(cb_pre_in2);  // configure unpack for c_1's data format
        copy_tile(cb_pre_in2, 0, 1);              // copy tile 0 from c_1 to DEST[1]

        // Unpack false-value tile from c_2 into DEST register 2
        copy_tile_to_dst_init_short(cb_pre_in3);  // configure unpack for c_2's data format
        copy_tile(cb_pre_in3, 0, 2);              // copy tile 0 from c_2 to DEST[2]

        // Initialize SFPU for where operation (configures SFPLOADMACRO instruction templates)
        TERNARY_SFPU_OP_INIT();   // expands to where_tile_init()
        // Execute the where operation: output = (DEST[0] != 0) ? DEST[1] : DEST[2], result in DEST[0]
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);  // expands to where_tile<DataFormat>(0, 1, 2, 0)

        // Signal that DEST registers are ready for packing
        tile_regs_commit();
        // Wait for pack engine to be ready
        tile_regs_wait();

        // Pack result from DEST[0] into output CB
        pack_tile(0, cb_out);

        // Release destination registers for next iteration
        tile_regs_release();

        // Push output tile to writer and free input tiles
        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in3, num_tiles_per_cycle);
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_where.h` (Wormhole B0)
`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h` (Blackhole)

#### Annotated SFPU Kernel Source (Wormhole B0)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_defs.h"
#include "lltt.h"   // lltt::record and lltt::replay for instruction replay buffer
#include "sfpi.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(
    const std::uint32_t dst_index_in0,   // DEST index for predicate (condition)
    const std::uint32_t dst_index_in1,   // DEST index for true-value
    const std::uint32_t dst_index_in2,   // DEST index for false-value
    const std::uint32_t dst_index_out)   // DEST index for output
{
    // Compile-time validation: only Float32, Float16_b, Int32, UInt32 supported
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b ||
        data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    // Calculate byte offsets into DEST register file for each tile face.
    // Each DEST tile occupies 32 rows, each row is 2 bytes wide in the SFPU address space.
    // The <<1 converts from row count to byte offset.
    int offset0 = (dst_index_in0 * 32) << 1;  // predicate offset in DEST
    int offset1 = (dst_index_in1 * 32) << 1;  // true-value offset in DEST
    int offset2 = (dst_index_in2 * 32) << 1;  // false-value offset in DEST

    // Select load/store modifier based on data format:
    // LO16 for bfloat16 (16-bit), INT32 for 32-bit formats (float32, int32, uint32)
    constexpr std::uint32_t mod0 =
        data_format == DataFormat::Float16_b ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

#ifdef DISABLE_SFPLOADMACRO
    // Fallback path: explicit instruction sequence without SFPLOADMACRO pipelining.
    // This is 6 instructions per face row, less efficient than the pipelined path.
    int offset3 = (dst_index_out * 32) << 1;

    lltt::record(0, 6);  // Record 6 instructions into replay buffer slot 0
    // Load predicate value from DEST[offset0] into SFPU local register LREG0
    TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset0);
    // Load true-value from DEST[offset1] into LREG1
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset1);
    // Set lane flags: for each of the 32 lanes, flag = (LREG0 == 0)
    // Lanes where predicate is zero will select the false-value
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    // Load false-value from DEST[offset2] into LREG1 (overwrites true-value only in flagged lanes)
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset2);
    // Enable condition codes: use lane flags for lane enable, then reset flags to true
    // This means: for lanes where predicate==0, LREG1 now has false-value;
    //             for lanes where predicate!=0, LREG1 still has true-value
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    // Store LREG1 (selected result) back to DEST[offset3]
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_6, offset3);

    // Replay the recorded 6 instructions for all 8 face rows (ITERATIONS=8)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        lltt::replay(0, 6);
    }
#else
    // Optimized path using SFPLOADMACRO for pipelined execution.
    // SFPLOADMACRO allows scheduling load + simple + store unit operations in parallel.
    if (dst_index_out == dst_index_in0)
    {
        // Special case: output overwrites the predicate register (where(a, b, c, a)).
        // Achieves 3 cycles per face row of 32 values using Macro 0 and Macro 2.
        //
        // Pipeline schedule per row:
        // Cycle | Load Unit               | Simple Unit                     | Store Unit
        // 0     | SFPLOAD L0=Dst[offset0] |                                 |
        // 1     | SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)  |
        // 2     | SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)      |
        // 3     | (next SFPLOAD L0)       |                                 | SFPSTORE Dst[offset0]=L0

        lltt::record(0, 3);  // Record 3 instructions into replay buffer
        // SFPLOADMACRO with macro 0 (bits 0:1 = macro index): loads predicate,
        // schedules SFPSETCC from instruction template 0 on the simple unit
        TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_7, offset0);
        // SFPLOADMACRO with macro 2: loads true-value,
        // schedules SFPENCC from instruction template 1 on the simple unit,
        // and schedules SFPSTORE on the store unit (from macro 0's store config)
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1);
        // Regular SFPLOAD: loads false-value (conditional write based on lane flags)
        TT_SFPLOAD(0, mod0, ADDR_MOD_6, offset2);

        // Replay for all 8 face rows
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 3);
        }
    }
    else
    {
        // General case: output goes to a different DEST register (where(a, b, c, d)).
        // Achieves 4 cycles per face row due to additional store instruction.
        //
        // Pipeline schedule per row:
        // Cycle | Load Unit               | Simple Unit                     | Store Unit
        // 0     | SFPLOAD L0=Dst[offset0] |                                 |
        // 1     | SFPLOAD L0=Dst[offset1] | SFPSETCC LaneEnabled=(L0 EQ 0)  |
        // 2     | SFPLOAD L0=Dst[offset2] | SFPENCC (LaneEnabled=true)      |
        // 3     | -                       |                                 | SFPSTORE Dst[offset3]=L0
        // 4     | (next SFPLOAD L0)       |                                 |

        int offset3 = (dst_index_out * 32) << 1;

        lltt::record(0, 4);  // Record 4 instructions into replay buffer
        TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_7, offset0);  // Macro 1: load predicate, schedule SFPSETCC
        TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1);  // Macro 2: load true-value, schedule SFPENCC
        TT_SFPLOAD(0, mod0, ADDR_MOD_7, offset2);               // Load false-value
        TT_SFPSTORE(0, mod0, ADDR_MOD_6, offset3);              // Store result to output offset

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 4);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void _init_where_()
{
#ifndef DISABLE_SFPLOADMACRO
    // Configure SFPLOADMACRO instruction templates for the where operation.

    // InstructionTemplate[0]: SFPSETCC with LREG_EQ0 mode
    // When triggered by a macro, this compares the previously loaded LREG0 value
    // against zero and sets per-lane flags accordingly.
    TTI_SFPSETCC(0, 0, 12, 6); // SFPSETCC_MOD1_LREG_EQ0

    // InstructionTemplate[1]: SFPENCC to enable conditional lane writes,
    // then reset lane flags to all-true for subsequent operations.
    TTI_SFPENCC(0, 0, 13, 0);

    // Macro 0: For the where(a, b, c, a) case (output == predicate register).
    // Configures simple unit to use template 0 (SFPSETCC) with delay=4,
    // and store unit to use template 2 (implicit) with delay=3.
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;  // template 0, delay 4
        constexpr std::uint32_t mad_bits    = 0;                             // no MAD unit instruction
        constexpr std::uint32_t round_bits  = 0;                             // no round unit instruction
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (2 << 3) | 3;  // template 2, delay 3

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);  // Write to macro slot 0
    }

    // Macro 1: For the where(a, b, c, d) case (output != predicate register).
    // Configures simple unit to use template 0 (SFPSETCC) with delay=4.
    // No store unit -- store is done explicitly via SFPSTORE instruction.
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4;  // template 0, delay 4
        constexpr std::uint32_t mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1);  // Write to macro slot 1
    }

    // Macro 2: Used by both cases.
    // Configures simple unit to use template 0 (SFPENCC -- actually template 1 based on context)
    // with delay=5.
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5;  // template 0, delay 5
        constexpr std::uint32_t mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1);  // Write to macro slot 2
    }

    // Misc configuration: UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1 for all macros.
    // This tells the hardware to use the load instruction's mod0 field for store operations
    // and to interpret delays as instruction counts rather than cycle counts.
    TTI_SFPCONFIG(0x770, 8, 1);
#endif
}

} // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` | Loads a 32-lane vector from DEST register file into an SFPU local register (LREG). Used to load predicate, true-value, and false-value tiles. |
| `SFPLOADMACRO` | Performs SFPLOAD plus schedules additional instructions on the simple/MAD/round/store sub-units in the same cycle, enabling pipelined execution. |
| `SFPSETCC` | Sets per-lane condition flags based on a comparison. Here uses `SFPSETCC_MOD1_LREG_EQ0`: flag = (LREG0 == 0). Lanes where predicate is zero are flagged. |
| `SFPENCC` | Controls lane-level predication. `SFPENCC_MOD1_EU_R1`: enables lane flags for subsequent operations, then resets flags to true. This causes the false-value load to only write to lanes where the predicate was zero. |
| `SFPSTORE` | Stores a 32-lane vector from an SFPU local register back to the DEST register file. Writes the selected result (true or false value per lane) to the output DEST offset. |
| `SFPLOADI` | Loads an immediate value into LREG0 for configuring SFPLOADMACRO instruction templates. |
| `SFPCONFIG` | Configures SFPLOADMACRO macro slots (defining which instruction templates execute on which sub-units and with what delays). |
| `lltt::record` / `lltt::replay` | Records a sequence of SFPU instructions into a replay buffer, then replays them for each of the 8 face rows (ITERATIONS=8) within a tile face. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Loaded with predicate value; used by SFPSETCC to test condition |
| **LREG1** | First loaded with true-value, then conditionally overwritten with false-value based on lane flags. Serves as the output register for SFPSTORE. |
| **DEST[0]** | Predicate tile (input, unpacked from c_0). Also used as output when `dst_index_out == dst_index_in0`. |
| **DEST[1]** | True-value tile (input, unpacked from c_1) |
| **DEST[2]** | False-value tile (input, unpacked from c_2) |
| **ADDR_MOD_7** | Address modifier with dest increment=0 (used for loads, no auto-increment) |
| **ADDR_MOD_6** | Address modifier with dest increment=2 (used for stores, advances to next face row pair) |
| **Lane Flags** | Per-lane boolean flags set by SFPSETCC and consumed by SFPENCC for conditional execution |

#### SFPU Execution Flow

1. **Tile Acquisition**: The compute kernel calls `cb_wait_front` on c_0 (predicate), c_1 (true-value), and c_2 (false-value) to ensure tiles are available from the reader.

2. **Unpack to DEST**: Three `copy_tile` calls unpack tiles from the circular buffers into DEST registers:
   - `copy_tile(c_0, 0, 0)` -- predicate to DEST[0]
   - `copy_tile(c_1, 0, 1)` -- true-value to DEST[1]
   - `copy_tile(c_2, 0, 2)` -- false-value to DEST[2]

3. **SFPU Initialization**: `where_tile_init()` configures the SFPU:
   - Sets up instruction templates (SFPSETCC with EQ0 mode, SFPENCC)
   - Configures three SFPLOADMACRO macros for pipelined execution
   - Configures address modifiers (ADDR_MOD_6 with dest increment=2 for store, ADDR_MOD_7 with no increment for loads)

4. **SFPU Math Operation**: `where_tile(0, 1, 2, 0)` calls `_calculate_where_` which processes all 4 faces of the tile (via `_llk_math_eltwise_ternary_sfpu_params_` which loops over 4 faces, calling the function for each). For each face:
   - The function processes 8 row iterations (ITERATIONS=8), each handling 32 lanes
   - For each row: SFPLOAD loads predicate into LREG0, SFPSETCC tests predicate==0, SFPLOAD loads true-value into LREG1, SFPLOAD conditionally loads false-value into LREG1 (only for lanes where predicate==0), SFPSTORE writes result
   - With SFPLOADMACRO pipelining, this achieves 3 cycles per row when output==predicate register, or 4 cycles otherwise

5. **Pack to Output CB**: After `tile_regs_commit()` and `tile_regs_wait()`, `pack_tile(0, c_3)` packs DEST[0] into the output circular buffer.

6. **Tile Release**: `cb_push_back(c_3, 1)` makes the output tile available to the writer, and `cb_pop_front` on all input CBs frees the consumed tiles.

#### SFPU Configuration

| Configuration | Value | Description |
|--------------|-------|-------------|
| **TERNARY_SFPU_OP_INIT** | `where_tile_init` | Macro define expanded by program factory |
| **TERNARY_SFPU_OP_FUNC** | `where_tile<DataFormat::Float16_b>` (BF16), `where_tile<DataFormat::Float32>` (FP32), `where_tile<DataFormat::Int32>` (INT32) | Templated on data format |
| **SfpuType** | `SfpuType::where` | Used for address modifier configuration |
| **fp32_dest_acc_en** | true for UINT32/INT32/FLOAT32 output | Enables 32-bit DEST accumulation |
| **UnpackToDestMode** | `UnpackToDestFp32` for FLOAT32 inputs, `Default` otherwise | Controls unpack precision |
| **SFPLOADMACRO** | Enabled by default, disabled with `DISABLE_SFPLOADMACRO` | Pipelined vs sequential execution |
| **FILL_LLK** | `fill_tile` (float), `fill_tile_int<DataFormat::Int32>` (int32), `fill_tile_uint<DataFormat::UInt32>` (uint32) | For TTS/TST scalar fill |

#### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations of `_calculate_where_` are functionally identical in their SFPU instruction sequence. The differences are:

1. **Address Modifiers**: Wormhole uses `ADDR_MOD_7` and `ADDR_MOD_6` while the same names map to different hardware addresses. Both configure the same logical behavior (no-increment for loads, increment-2 for stores).

2. **Replay Buffer API**: Blackhole uses `load_replay_buf(slot, count, lambda)` which is a higher-level wrapper around `lltt::record`/`lltt::replay`. Wormhole uses the raw `lltt::record` + explicit instructions + `lltt::replay` pattern.

3. **SFPLOADMACRO**: Both architectures support SFPLOADMACRO with the same macro configuration format. However, Blackhole does NOT automatically stall for data hazards within SFPLOADMACRO sequences, so manual scheduling (delay values in macro config) remains critical on both platforms.

4. **The `_init_where_` function** is identical for both architectures, configuring the same instruction templates and macros.

## Implementation Notes

1. **NaN Handling**: There is a commented-out hack in the program factory (lines 828-834) suggesting that bfloat16 CBs could be configured as UINT16 to preserve NaN values. Currently, bfloat16 packs NaN as infinity due to a known hardware limitation.

2. **Broadcast Complexity**: The operation supports 7 broadcast types across 3 variants, resulting in 18 distinct kernel configurations for WHERE alone (as enumerated in `kernel_config_map`). The broadcast detection logic considers all three input shapes independently.

3. **Program Caching**: The `override_runtime_arguments` method enables efficient re-execution when only tensor addresses change. It re-detects the broadcast type and updates all per-core runtime arguments without recreating the program.

4. **Width Sharding Support**: Both reader and writer kernels support width-sharded tensors by limiting the inner W-tile loop to `dst_shard_width` and adjusting tile offsets accordingly. The reader uses `end_tw = has_sharding ? start_tw + dst_shard_width : Wt`.

5. **SFPLOADMACRO Optimization**: The where operation is highly optimized using SFPLOADMACRO to pipeline load/compare/store across SFPU sub-units. The in-place case (output == predicate) achieves 3 cycles per row (96 cycles per tile face), while the general case takes 4 cycles per row (128 cycles per tile face). Since the compute kernel always calls `where_tile(0, 1, 2, 0)` where `dst_index_out=0` equals `dst_index_in0=0`, the 3-cycle optimized path is always taken.

6. **ScalarVariant Packing**: For TTS/TST variants, the scalar value is packed into a uint32_t at the host side and transmitted as a runtime argument. The compute kernel reinterpret-casts it back to float or uses it directly as int32/uint32, then fills an entire DEST register with `fill_tile`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the ternary eltwise operation (like 'where') work in TTNN?"
   **Reason**: Initial architectural understanding of the operation structure, kernel selection, and program factory organization.
   **Key Findings**: Confirmed the TernaryProgramFactory structure, kernel_config_map-based kernel selection, and the TTT/TTS/TST variant system. Identified that the low-level kernel is `llk_math_eltwise_ternary_sfpu_where`.

2. **Query**: "What is the SFPU 'where' operation? How does it work at the compute kernel level?"
   **Reason**: Understanding the SFPU-level implementation before reading the actual kernel source.
   **Key Findings**: Confirmed the `_calculate_where_` function as the core SFPU kernel, identified the use of SFPSETCC for condition evaluation and SFPENCC for lane-level predication. Learned about the `v_if`/`v_endif` abstraction layer (though where uses raw instructions instead).

3. **Query**: "What are SFPSETCC and SFPENCC SFPU instructions? How does SFPLOADMACRO enable pipelined execution?"
   **Reason**: Deep understanding of the SFPU instructions used in the where kernel.
   **Key Findings**: SFPSETCC sets per-lane flags based on comparison (EQ0 mode used here). SFPENCC controls whether lane flags gate subsequent writes. SFPLOADMACRO pipelines load + up to 4 other sub-unit operations in parallel, configured via SFPCONFIG with instruction templates and delay values.

4. **Query**: "How does the SFPU condition code system work? What do SFPSETCC and SFPENCC do?"
   **Reason**: Cross-referencing SFPU condition code behavior from the SFPI perspective.
   **Key Findings**: Confirmed 32-lane (per face row) condition code system. SFPSETCC_MOD1_LREG_EQ0 sets flag when value equals zero. SFPENCC_MOD1_EU_R1 enables lane flags and resets result to true. The `enabled(lane)` method returns `!enable[lane] || result[lane]`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp`
   **Reason**: Understanding kernel file path mapping and broadcast type detection.
   **Key Information**: Complete `kernel_config_map` with 18 WHERE entries, `get_kernel_file_path` mapping from KernelName enum to file paths, broadcast detection logic for 2-tensor and 3-tensor cases.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu_params.h`
   **Reason**: Understanding how the ternary SFPU function is dispatched across tile faces.
   **Key Information**: `_llk_math_eltwise_ternary_sfpu_params_` iterates over 4 faces in VectorMode::RC (default), calling the SFPU function for each face and advancing the DEST pointer by 2 rows (16 elements) between faces via `TTI_SETRWC`.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu.h`
   **Reason**: Understanding SFPU initialization and address modifier configuration.
   **Key Information**: `_llk_math_eltwise_ternary_sfpu_init_` configures ADDR_MOD_7 (dest increment=0 for loads) and ADDR_MOD_6 (dest increment=2 for stores, specific to where operation). `_llk_math_eltwise_ternary_sfpu_start_` stalls until SFPU is ready, `_done_` waits for SFPU completion.
