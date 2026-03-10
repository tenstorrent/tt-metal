# MAXIMUM (element_wise_multi_core_sfpu) Implementation Analysis

## Overview
The MAXIMUM operation computes the element-wise maximum of two input tensors: `y = max(x0, x1)`. It is a binary SFPU operation dispatched through the `ElementWiseMultiCoreSfpu` program factory. The operation supports three data-type variants: floating-point (default), INT32, and UINT32, each selecting a different SFPU kernel path.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition
One work unit is one 32x32 tile. Tiles are grouped into blocks of size `max_block_size` (the largest power-of-two divisor of `num_tiles_per_shard` when sharded, otherwise 1). The compute kernel processes `per_core_block_cnt` blocks, each containing `per_core_block_size` tiles.

## Tensor Format and Layout

### Input Tensors

| Property | Input A (src0) | Input B (src1) |
|---|---|---|
| Dimension Convention | NHWC (logical), row-major tile order | NHWC (logical), row-major tile order |
| Tensor Layout | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | BFLOAT16 / FLOAT32 / INT32 / UINT32 | BFLOAT16 / FLOAT32 / INT32 / UINT32 |

If sharded, shard parameters are derived from whichever input (or output) is sharded. Shard shape, core grid, and orientation come from the tensor's `shard_spec`.

### Output Tensor

| Property | Output |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | Same as determined by operation attributes |

### Layout Transformations
No tilize/untilize or format conversions are performed. All tensors must already be in tiled layout. When inputs and output use different data formats, `UnpackToDestMode::UnpackToDestFp32` is used (for non-POWER ops) to unpack tiles into FP32 in the DEST register, and the packer converts back on output.

## Data Flow Pattern

1. **Reader kernel** reads tiles from input A (`src0`) into `cb_in0` (CB c_0) and from input B (`src1`) into `cb_in1` (CB c_1). If an input is sharded, the reader simply does `cb_reserve_back` / `cb_push_back` to make the pre-existing L1 data visible to the consumer. Otherwise it issues `noc_async_read_tile` from DRAM.
2. **Compute kernel** waits for tiles in `cb_inp0` and `cb_inp1` (which equal `cb_in0`/`cb_in1` for MAXIMUM since no pre-scaling is needed), copies them into DEST registers (input A at even slots `i*2`, input B at odd slots `i*2+1`), executes `binary_max_tile(i*2, i*2+1, i*2)` which runs the SFPU SWAP min/max, then packs the result from DEST slot `i*2` into `cb_out0` (CB c_2).
3. **Writer kernel** reads completed tiles from `cb_out0` and writes them to the output buffer via `noc_async_write_page`. If output is sharded, it simply waits on `cb_out0` since the CB is backed by the output buffer directly.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (reader pushes 1 tile at a time for interleaved) | Double-buffered (interleaved) or full-shard (sharded) | Reader | Compute | Full program |
| c_1 | cb_src1 | Input B tiles | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (reader pushes 1 tile at a time for interleaved) | Double-buffered (interleaved) or full-shard (sharded) | Reader | Compute | Full program |
| c_2 | cb_output | Output tiles | Sharded/block-width: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (writer pops 1 tile at a time for interleaved) | Double-buffered (interleaved) or full-shard (sharded) | Compute | Writer | Full program |

Note: For MAXIMUM, no interim CBs (c_3, c_4) are created because the operation does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`.

## Pipeline Pattern Summary
- **Interleaved path**: CBs c_0, c_1, and c_2 each have capacity `2 * max_block_size`. With `max_block_size = 1` (typical for interleaved), this yields 2-tile capacity, enabling **double-buffering** -- the reader can fill the next tile while compute processes the current one.
- **Sharded path**: CBs hold all shard tiles at once (single-buffered with respect to the full shard), but the entire shard is available immediately since the CB is backed by the tensor's L1 allocation.

## Index Calculations
- **Interleaved**: The reader iterates tile IDs from `start_id` to `start_id + num_tiles`. Each tile ID maps to a physical page via `TensorAccessor` which handles bank interleaving. The writer uses the same scheme.
- **Block/width sharded**: The reader computes `start_id` based on the core's position in the shard grid: `start_id = (core_idx / num_shards_per_width) * (block_height * block_width * num_shards_per_width) + (core_idx % num_shards_per_width) * block_width`. It then iterates rows with stride `num_cores_y * block_width`.
- **DEST register indexing**: Input A tile `i` is placed at DEST slot `i*2`, input B at `i*2+1`. The SFPU operates on `(i*2, i*2+1)` and writes the result to `i*2`. Pack reads from `i*2`.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile-by-tile reads via NoC. One tile at a time with `noc_async_read_barrier` after each pair of reads (src0 + src1).
- **Sharded**: No NoC reads; data is already in L1. The reader just marks the CB as available.

### Write Pattern
- **Interleaved**: Sequential page-by-page writes via NoC with `noc_async_writes_flushed` after each tile and a final `noc_async_write_barrier`.
- **Sharded**: No NoC writes; the output CB directly aliases the output tensor buffer in L1.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Determined by `operation_attributes.worker_grid` (a `CoreRangeSet`) |
| Work Splitting | `split_work_to_cores` divides total tiles evenly across cores |
| Core Group 1 | Cores with `ceil(num_tiles / num_cores)` tiles |
| Core Group 2 | Cores with `floor(num_tiles / num_cores)` tiles (remainder handling) |
| Sharded | Each core processes its own shard; all cores in core_group_1 with `num_tiles_per_shard` tiles |
| Row Major | Default for interleaved; shard orientation for sharded |
| Unused Cores | Runtime args zeroed out (num_tiles = 0) |

The factory supports an optimization for "zero-start grids" (single rectangular CoreRange starting at (0,0)) which uses faster work-splitting algorithms.

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if any tensor uses block or width sharding, 0 otherwise |
| 1+ | TensorAccessorArgs (src0) | variable | Tensor accessor args for input A (only if not IN0_SHARDED) |
| N+ | TensorAccessorArgs (src1) | variable | Tensor accessor args for input B (only if not IN1_SHARDED) |

**Reader defines:**
- `IN0_SHARDED`: defined if input A is sharded
- `IN1_SHARDED`: defined if input B is sharded

**Writer kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs (dst) | variable | Tensor accessor args for output buffer |

**Writer defines:**
- `OUT_SHARDED`: defined if output is sharded

**Compute kernel:**

No integer compile-time args. Configuration is via defines:

| Define | Value (for float MAXIMUM) | Description |
|---|---|---|
| `BINOP_INIT` | `binary_max_tile_init();` | SFPU init function call |
| `BINARY_SFPU_OP` | `binary_max_tile(i*2, i*2+1, i*2);` | Per-tile SFPU operation |

For INT32: `BINOP_INIT` = `binary_max_int32_tile_init();`, `BINARY_SFPU_OP` = `binary_max_int32_tile(i*2, i*2+1, i*2);`
For UINT32: `BINOP_INIT` = `binary_max_uint32_tile_init();`, `BINARY_SFPU_OP` = `binary_max_uint32_tile(i*2, i*2+1, i*2);`

ComputeConfig also sets:
- `fp32_dest_acc_en`: true if output is Float32, Int32, or UInt32
- `unpack_to_dest_mode`: `UnpackToDestFp32` for all CBs (since op is not POWER)

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | DRAM/L1 address of input A buffer |
| 1 | src1_addr | uint32_t | DRAM/L1 address of input B buffer |
| 2 | num_tiles | uint32_t | Number of tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if not sharded) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if not sharded) |
| 6 | num_cores_y | uint32_t | Number of shards per width dimension |

**Compute kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks this core processes |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

**Writer kernel (interleaved path):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles this core writes |
| 2 | start_id | uint32_t | Starting tile ID for writes |

## Kernel Implementations

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Reads pairs of tiles (one from each input) sequentially. For sharded inputs, simply marks the CB as populated. For block/width-sharded non-sharded inputs, uses a 2D loop (block_height x block_width) with row stride `num_cores_y * block_width`. Issues `noc_async_read_barrier` after each tile pair to ensure data arrives before signaling the compute kernel.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Standard unary writer. Iterates from `start_id` to `start_id + num_pages`, waiting for one tile in `cb_out`, reading the L1 pointer, writing it to DRAM via `noc_async_write_page`, flushing, then popping. For sharded output, just waits on all pages (no writes needed).

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
#include "api/compute/binary_max_min.h"       // provides binary_max_tile, binary_max_tile_init, etc.
#include "api/compute/xlogy.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/binary_comp.h"

// PRE_SCALE is true if pre-scaling defines exist; for MAXIMUM this is false
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // number of blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // input A circular buffer
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // input B circular buffer

    // For MAXIMUM, SFPU_OP_INIT_PRE_IN0_0 is NOT defined, so cb_inp0 = cb_in0
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;
#else
    constexpr auto cb_inp0 = cb_in0;             // no pre-scaling, use input CB directly
#endif

    // For MAXIMUM, SFPU_OP_INIT_PRE_IN1_0 is NOT defined, so cb_inp1 = cb_in1
#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;
#else
    constexpr auto cb_inp1 = cb_in1;             // no pre-scaling, use input CB directly
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // output circular buffer

    unary_op_init_common(cb_in0, cb_out0);        // initialize unpack/pack pipeline for cb_in0 -> cb_out0

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));   // not active for MAXIMUM unless fused
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // PRE_SCALE block: skipped for MAXIMUM (no pre-scaling defines)
#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

        // SFPU_OP_INIT_PRE_IN0_0 block: skipped for MAXIMUM
#ifdef SFPU_OP_INIT_PRE_IN0_0
        // ... pre-scale input A ...
#endif

        // SFPU_OP_INIT_PRE_IN1_0 block: skipped for MAXIMUM
#ifdef SFPU_OP_INIT_PRE_IN1_0
        // ... pre-scale input B ...
#endif

        // Wait for both inputs to be available
        cb_wait_front(cb_inp0, per_core_block_size);   // blocks until reader has produced tiles in cb_in0
        cb_wait_front(cb_inp1, per_core_block_size);   // blocks until reader has produced tiles in cb_in1
        cb_reserve_back(cb_out0, per_core_block_size); // reserve space in output CB for compute results

        tile_regs_acquire();                           // acquire exclusive access to DEST registers
        tile_regs_wait();                              // wait for DEST registers to be available

        // Copy all input A tiles into even DEST slots
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);  // configure unpack for cb_inp0's data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);              // unpack tile i from cb_inp0 into DEST[i*2]
        }

        // Copy all input B tiles into odd DEST slots, then apply SFPU op per tile
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);  // reconfigure unpack for cb_inp1's data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);         // unpack tile i from cb_inp1 into DEST[i*2+1]

            // For MAXIMUM, BINOP_INIT expands to: binary_max_tile_init();
#ifdef BINOP_INIT
            BINOP_INIT                                  // initialize SFPU for max operation (configures SFPSWAP, SFPLOADMACRO)
#endif

            // For MAXIMUM, BINARY_SFPU_OP expands to: binary_max_tile(i*2, i*2+1, i*2);
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP                              // execute SFPU max: result of max(DEST[i*2], DEST[i*2+1]) -> DEST[i*2]
#endif

            // SFPU_OP_INIT_0 and SFPU_OP_CHAIN_0: not defined for MAXIMUM (no fused post-ops by default)
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            pack_tile(i * 2, cb_out0);                 // pack DEST[i*2] (the max result) into output CB
        }
        tile_regs_commit();                            // signal that DEST register writes are complete
        tile_regs_release();                           // release DEST registers for next iteration

        cb_pop_front(cb_inp0, per_core_block_size);    // free consumed input A tiles
        cb_pop_front(cb_inp1, per_core_block_size);    // free consumed input B tiles
        cb_push_back(cb_out0, per_core_block_size);    // publish output tiles to writer kernel
    }
}
```

### SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
- **API header**: `tt_metal/hw/inc/api/compute/binary_max_min.h`
- **LLK dispatch**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h`
- **SFPU kernel**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`

Where `{arch}` is `wormhole_b0` or `blackhole`.

#### Annotated SFPU Kernel Source (Wormhole B0)
```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// ============================================================
// FLOATING-POINT MAX/MIN
// ============================================================

// calculate_binary_max_min<IS_MAX_OP=true> computes element-wise maximum.
// Called with ITERATIONS=8 (default) which processes all 8 row-groups
// within one face of a 16x16 sub-tile (each face has 16 rows, processed
// in groups of 2 rows = 8 iterations).
template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Convert tile indices to DEST register byte offsets.
    // Each tile occupies 32 rows in DEST, each row is 2 bytes wide in the
    // SFPU's addressing scheme, hence (index * 32) << 1.
    uint offset0 = (dst_index_in0 * 32) << 1;   // byte offset for input A in DEST
    uint offset1 = (dst_index_in1 * 32) << 1;   // byte offset for input B in DEST
    uint offset2 = (dst_index_out * 32) << 1;   // byte offset for output in DEST

#ifdef DISABLE_SFPLOADMACRO
    // Fallback path without SFPLOADMACRO -- simpler but slower (4 cycles/row).
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load row from input A into LREG0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0);
        // Load row from input B into LREG1
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        // SFPSWAP with VEC_MIN_MAX mode: after execution, LREG1 = max, LREG0 = min
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        // Store the maximum (LREG1) or minimum (LREG0) based on IS_MAX_OP template param
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2);
    }
#else
    // Optimized path using SFPLOADMACRO for 3-cycle-per-row throughput.
    //
    // SFPLOADMACRO schedules loads and instruction templates across pipeline stages.
    // The pipeline diagram shows how load, swap, round (convert), and store overlap:
    //
    // t | Load | Simple              | MAD | Round     | Store   |
    // - | ---- | ------------------- | --- | --------- | ------- |
    // 0 | [a]  |                     |     |           |         |   -- load input A via macro
    // 1 |  b   |                     |     |           |         |   -- load input B directly
    // 2 | [c]  | swap_minmax([a], b) |     |           |         |   -- store macro + swap in simple stage
    // 0 | ...  |                     |     |           |         |
    // 1 | ...  |                     |     | L16 = [a] |         |   -- round stage converts result
    // 2 | ...  |                     |     |           | [c] L16 |   -- store stage writes to DEST

    constexpr int b = p_sfpu::LREG2;   // scratch register for input B
    constexpr int c = p_sfpu::LREG3;   // scratch register for store macro

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate LREG0 and LREG1 to avoid pipeline stalls
        // Macro 0: load input A row into alternating LREG, triggers swap template in simple stage
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0 | (a >> 2));
        // Direct load of input B row into LREG2
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        // Macro 1: triggers store template, writing result to output DEST location
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2 | (c >> 2));
    }

    // Pipeline drain: 3 NOPs to flush remaining in-flight operations
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

// ============================================================
// INTEGER MAX/MIN (INT32 / UINT32)
// ============================================================

// calculate_binary_max_min_int32 handles signed and unsigned 32-bit integer max/min.
// The SFPSWAP instruction operates on floating-point representation, so for integers
// a correction step using SFPSETCC + conditional SFPSWAP is needed.
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load integers from DEST into LREG0 and LREG1
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, offset1);
        // SFPSWAP with VEC_MIN_MAX: sorts by FP magnitude, but INT32 sign encoding differs
        // For unsigned: mod1=9 swaps to put max in VD; for signed: SFPSWAP_MOD1_VEC_MIN_MAX
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // Correction for integer comparison: SFPSWAP operates on FP exponent/mantissa
        // which doesn't correctly order negative integers. The condition code tests
        // fix cases where the swap produced the wrong ordering.
        // For unsigned: test >=0 (always true for uint, but needed for sign-magnitude edge cases)
        // For signed: test <0 to detect negative numbers that were mis-ordered
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        // Conditionally swap LREG0 and LREG1 based on the condition codes set above
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        // Clear condition codes
        TTI_SFPENCC(0, 0, 0, 0);

        // Store the correct result (max or min) back to DEST
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, offset2);
    }
#else
    // Optimized SFPLOADMACRO path for integers: 5 cycles per row due to extra
    // SFPSETCC/SFPENCC correction steps.
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

    // Record 10 instructions into replay buffer starting at index 0
    load_replay_buf(0, 10, [offset0, offset1, offset2] {
        // First iteration uses a0, b0, c
        TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a0 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b0 >> 2));
        TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));

        // Second iteration uses a1, b1, c
        TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a1 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b1 >> 2));
        TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));
    });

    // Replay the 10-instruction sequence ITERATIONS/2 times (processes 2 rows per replay)
#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);
    }

    // Handle odd iteration count and drain the pipeline
    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);   // replay the SFPSETCC+SFPENCC drain
    }

    TTI_SFPNOP;   // final pipeline drain
#endif
}

// ============================================================
// INIT FUNCTIONS
// ============================================================

// binary_max_min_init configures SFPLOADMACRO instruction templates and macros
// for the floating-point max/min path.
template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]: SFPSWAP that sorts LREG values by min/max
    // mod1=9 means VD gets max, VC gets min (when IS_MAX_OP=true)
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[1]: SFPSHFT2 used as a no-op conversion/passthrough for store
    TTI_SFPSHFT2(0, 0, 13, 6);   // SFPSHFT2_MOD1_SHFT_IMM -- shift by immediate 0

    // Macro 0: defines the pipeline schedule for load+swap+round stages
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;   // use template[0] in simple stage, advance DEST ptr
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;   // use template[1] in round stage for format conversion
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);   // program macro 0
    }

    // Macro 1: defines the store stage pipeline schedule
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;   // store from round result

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);   // program macro 1
    }

    // Misc configuration: StoreMod0=DEFAULT, UsesLoadMod0ForStore={1,1}, UnitDelayKind={1,1}
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}

// binary_max_min_int32_init configures SFPLOADMACRO for integer max/min,
// including the additional SFPSETCC correction templates.
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int b1 = p_sfpu::LREG3;

    // InstructionTemplate[0]: SFPSWAP for first iteration pair
    TTI_SFPSWAP(0, b0, 12, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[1]: SFPSWAP for second iteration pair
    TTI_SFPSWAP(0, b1, 13, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[2]: SFPSETCC for condition code correction
    TTI_SFPSETCC(0, 0, 14, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);

    // InstructionTemplate[3]: SFPSHFT2 passthrough for store
    TTI_SFPSHFT2(0, 0, 15, 6);

    // Macros 0-3 define the 5-stage pipeline for integer max/min
    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (4 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Macro 3
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: StoreMod0=DEFAULT, UsesLoadMod0ForStore={1,1,1,1}, UnitDelayKind={1,1,1,1}
    TTI_SFPCONFIG(0xff0, 8, 1);
#endif
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|---|---|
| `SFPLOAD` / `TT_SFPLOAD` | Loads a row of data from DEST register into an SFPU local register (LREG). Supports DEFAULT (float) and INT32 modes. |
| `SFPSTORE` / `TT_SFPSTORE` | Stores a row of data from an SFPU local register back to DEST. Supports DEFAULT and INT32 modes. |
| `SFPSWAP` / `TTI_SFPSWAP` | Swaps values between two LREGs. With `SFPSWAP_MOD1_VEC_MIN_MAX` mode, it simultaneously sorts elements so VD=max and VC=min. With `SFPSWAP_MOD1_SWAP` mode, performs conditional swap based on condition codes. |
| `SFPSETCC` / `TTI_SFPSETCC` | Sets condition codes per-element based on LREG values. `SFPSETCC_MOD1_LREG_LT0` tests if value < 0; `SFPSETCC_MOD1_LREG_GTE0` tests if value >= 0. Used for integer correction logic. |
| `SFPENCC` / `TTI_SFPENCC` | Clears (disables) condition codes, returning to unconditional execution. |
| `SFPLOADMACRO` / `TT_SFPLOADMACRO` | Triggers a pre-programmed macro that schedules load + instruction template execution across multiple pipeline stages, enabling high-throughput pipelined execution. |
| `SFPLOADI` / `TTI_SFPLOADI` | Loads an immediate value into LREG0 (used during init to program macro bit-fields). |
| `SFPCONFIG` / `TTI_SFPCONFIG` | Configures SFPU macro slots and misc settings (store mode, delay kind). |
| `SFPSHFT2` / `TTI_SFPSHFT2` | Shift instruction used as a passthrough/conversion step in the pipeline template. |
| `SFPNOP` / `TTI_SFPNOP` | No-operation; used to drain the SFPU pipeline after the last iteration. |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `LREG0` | Input A row data (alternates with LREG1 in SFPLOADMACRO path) |
| `LREG1` | Input B row data (fallback path) or alternate input A (SFPLOADMACRO path) |
| `LREG2` | Input B row data (SFPLOADMACRO float path) or alternate pair register (int path) |
| `LREG3` | Store macro target (SFPLOADMACRO float path) or alternate pair register (int path) |
| `LREG7` | Store macro target (SFPLOADMACRO int path, Blackhole only) |
| DEST registers | Tile data: even slots (i*2) hold input A / output, odd slots (i*2+1) hold input B |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` and `tile_regs_wait()` to gain exclusive access to the DEST register file.
2. **Unpack input A**: `copy_tile(cb_inp0, i, i*2)` unpacks tile `i` from input A's circular buffer into DEST slot `i*2`. This uses the unpack engine (not SFPU).
3. **Unpack input B**: `copy_tile(cb_inp1, i, i*2+1)` unpacks tile `i` from input B's circular buffer into DEST slot `i*2+1`.
4. **SFPU init**: `binary_max_tile_init()` programs the SFPLOADMACRO instruction templates and macros. For the float path, this sets up Template[0] as SFPSWAP(VEC_MIN_MAX) and Template[1] as SFPSHFT2 (passthrough), then configures two macros for the 3-cycle pipelined schedule.
5. **SFPU compute**: `binary_max_tile(i*2, i*2+1, i*2)` calls through the LLK dispatch:
   - `_llk_math_eltwise_binary_sfpu_params_` iterates over all 4 faces of the tile (VectorMode::RC), calling `calculate_binary_max_min<true>` for each face.
   - Within each face, 8 iterations process all 16 rows (2 rows per iteration due to DEST addressing).
   - Each iteration: SFPLOAD reads a row-pair from input A and input B in DEST into LREGs, SFPSWAP sorts them into max/min, SFPSTORE writes the max back to the output DEST location.
   - Between faces, `TTI_SETRWC` advances the DEST row pointer by 16 rows (8 rows x 2 advances).
6. **Pack output**: `pack_tile(i*2, cb_out0)` reads the result from DEST slot `i*2` and packs it into the output circular buffer. The pack engine handles format conversion (e.g., FP32 to BFLOAT16).
7. **Release**: `tile_regs_commit()` and `tile_regs_release()` free the DEST registers. `cb_pop_front` / `cb_push_back` manage circular buffer flow.

#### SFPU Configuration
- **Math fidelity**: Not directly applicable (SFPU max uses comparison, not approximation).
- **APPROX template parameter**: Passed through the LLK chain but does not affect the max/min logic.
- **`fp32_dest_acc_en`**: Enabled when output dtype is Float32/Int32/UInt32, allocating full 32-bit precision in DEST.
- **`UnpackToDestMode::UnpackToDestFp32`**: Forces unpacking to FP32 format in DEST for all non-POWER binary SFPU ops, ensuring full precision for the SFPSWAP comparison.
- **SFPLOADMACRO configuration**: The init function programs 2 macros (float) or 4 macros (int) that define the pipeline schedule. Each macro encodes which instruction template to use in which pipeline stage (simple, MAD, round, store).

#### Hardware Compatibility Notes
- **Wormhole B0 vs Blackhole**: Both architectures use identical SFPU kernel logic. The only difference is in address modifier constants (`ADDR_MOD_3`/`ADDR_MOD_2` for Wormhole vs `ADDR_MOD_7`/`ADDR_MOD_6` for Blackhole) used in SFPLOAD/SFPSTORE instructions, reflecting different DEST address modifier register configurations.
- **Blackhole integer path**: Uses `load_replay_buf` lambda API for recording replay buffer instructions (more modern API), while Wormhole uses `lltt::record<lltt::NoExec>` for the same purpose.
- **SFPLOADMACRO**: Available on both architectures. The `DISABLE_SFPLOADMACRO` fallback path exists for testing/debugging but is not used in production.

## Implementation Notes

1. **No pre-scaling for MAXIMUM**: Unlike operations such as LOGADDEXP that apply EXP to inputs before combining, MAXIMUM has no pre-scaling step. The `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` defines are not set, so interim CBs (c_3, c_4) are not created and the pre-scale code blocks are compiled out.

2. **Per-tile init call**: The `BINOP_INIT` macro (`binary_max_tile_init()`) is called inside the per-tile loop, which means the SFPLOADMACRO configuration is re-programmed for every tile. This is a known pattern in the binary SFPU kernel -- the init is placed after `copy_tile` for input B to ensure the SFPU pipeline is correctly configured before the operation executes. While seemingly redundant, this is necessary because other operations in the same kernel template may reconfigure the SFPU between tiles.

3. **DEST slot interleaving**: Input A tiles occupy even DEST slots (`i*2`) and input B tiles occupy odd slots (`i*2+1`). The output overwrites the input A slot (`i*2`), which is also the slot passed to `pack_tile`. This means input A data is destroyed after the SFPU operation.

4. **Integer correction logic**: The `SFPSWAP(VEC_MIN_MAX)` instruction compares values using floating-point magnitude comparison. For signed integers stored in sign-magnitude or two's complement format, this can produce incorrect results for negative numbers. The `SFPSETCC` + conditional `SFPSWAP` sequence corrects these cases, adding 2 extra cycles per row for the integer path (5 cycles/row vs 3 cycles/row for float).

5. **Fused activations**: The operation supports optional fused post-processing via `SFPU_OP_CHAIN_0` and `SFPU_OP_INIT_0` defines, as well as `PACK_RELU` for fused ReLU. These are configured through the `fused_activations` parameter in the operation attributes.

## External Knowledge Sources

### DeepWiki Queries
1. **Query**: "How does the element_wise_multi_core_sfpu binary program factory work?"
   **Reason**: Needed to understand the overall structure, kernel selection, CB configuration, and work distribution strategy before reading source code.
   **Key Findings**: Confirmed the three-kernel architecture (reader/compute/writer), CB indices, sharded vs interleaved paths, and that `set_eltwise_binary_runtime_args` handles work distribution.

2. **Query**: "What does _llk_math_eltwise_binary_sfpu_params_ do in tt-llk?"
   **Reason**: The submodule was not checked out, so could not read the source directly. Needed to understand how the SFPU function is dispatched across tile faces.
   **Key Findings**: The function iterates over 4 faces in VectorMode::RC, calling the SFPU function for each face with TTI_SETRWC to advance the DEST pointer between faces. It wraps the call with `_llk_math_eltwise_binary_sfpu_start_` and `_llk_math_eltwise_binary_sfpu_done_`.

### Confluence References
Not consulted for this analysis. The SFPU instructions used (SFPSWAP, SFPLOAD, SFPSTORE, SFPSETCC, SFPENCC, SFPLOADMACRO) were sufficiently documented in the source code comments and DeepWiki.

### Glean References
Not consulted for this analysis.

### Documentation References
1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 334-345)
   **Reason**: Needed to determine exactly which defines are set for MAXIMUM and how the `op_name` maps to SFPU function calls.
   **Key Information**: MAXIMUM sets `BINOP_INIT` = `binary_max_tile_init()` and `BINARY_SFPU_OP` = `binary_max_tile(i*2, i*2+1, i*2)` for float. INT32 and UINT32 variants use `binary_max_int32_tile` and `binary_max_uint32_tile` respectively.

2. **Source**: `tt_metal/hw/inc/api/compute/binary_max_min.h`
   **Reason**: API header showing the ckernel-level function signatures and documentation for all max/min variants.
   **Key Information**: `binary_max_tile(idst0, idst1, odst)` dispatches to `llk_math_eltwise_binary_sfpu_binary_max<APPROX>`. DST buffer supports up to 4 tiles per operand for 16-bit formats, 2 for 32-bit.

3. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp`
   **Reason**: Contains `set_eltwise_binary_runtime_args` which handles all runtime argument setup and work distribution logic.
   **Key Information**: Work split uses `split_work_to_cores` for interleaved, shard grid for sharded. Supports zero-start-grid optimization. Block size determined by `find_max_block_size`.
