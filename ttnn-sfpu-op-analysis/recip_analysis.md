# RECIP (Reciprocal) Implementation Analysis

## Overview

The RECIP operation computes the element-wise reciprocal (`1/x`) of each element in the input tensor. It is implemented as a unary SFPU operation using the shared `UnaryProgramFactory` in `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`. The SFPU kernel uses Newton-Raphson refinement on a hardware-accelerated initial approximation (Blackhole) or a polynomial initial estimate (Wormhole B0) to achieve the desired precision.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

One work unit is **one tile** (32x32 elements). The program factory divides the total number of tiles across cores, and each core processes its assigned tiles one at a time (block size = 1 tile). The compute kernel processes `per_core_block_cnt` blocks, each containing `per_core_block_dim` = 1 tile.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| Dimension Convention | NHWC (logical), row-major page order |
| Tensor Layout | TILE (32x32) or ROW_MAJOR |
| Memory Layout | Interleaved across DRAM banks |
| Buffer Type | DRAM (interleaved) |
| Data Type | Float32, Float16_b, Bfp8_b (full accuracy); other float types supported |

### Output Tensor

| Property | Value |
|---|---|
| Dimension Convention | Same as input |
| Tensor Layout | Same as input |
| Memory Layout | Interleaved across DRAM banks |
| Buffer Type | DRAM (interleaved) |
| Data Type | Same as input (may differ if output dtype is specified) |

### Layout Transformations

No tilize/untilize or reshard operations are performed within the program factory. Input and output are expected in the same layout.

## Data Flow Pattern

1. **Reader kernel** reads one tile at a time from DRAM into circular buffer `c_0` (input CB) using `noc_async_read_page`.
2. **Compute kernel** waits for a tile in `c_0`, copies it to DST registers via `copy_tile`, executes the SFPU reciprocal chain (`recip_tile_init` + `recip_tile`), then packs the result into circular buffer `c_2` (output CB).
3. **Writer kernel** waits for a tile in `c_2`, reads it from L1, and writes it to DRAM using `noc_async_write_page`.

The flow is strictly sequential per tile: Reader -> Compute -> Writer. Double-buffering on the CBs allows the reader to fill the next slot while compute processes the current one.

## Circular Buffer Configuration

| CB ID | Index | Purpose | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| c_0 | 0 | Input tiles from DRAM | `single_tile_size` (input dtype) | 2 | 2 x tile_size | Double | Reader | Compute |
| c_2 | 2 | Output tiles to DRAM | `single_tile_size_output` (output dtype) | 2 | 2 x tile_size | Double | Compute | Writer |

Note: CB `c_1` (tmp0) is only created for HARDSHRINK or LOGIT operations, not for RECIP.

## Pipeline Pattern Summary

Both input and output CBs have capacity for 2 pages with a block size of 1 page, enabling **double-buffered** operation. This allows the reader to produce the next tile into CB `c_0` while the compute kernel is processing the current tile, and similarly the compute kernel can produce into `c_2` while the writer drains the previous result.

## Index Calculations

Tile indices are mapped to physical DRAM pages using the `TensorAccessor` utility. The reader and writer kernels receive a `start_id` (tile offset) and `num_pages` count. For each core, tiles are accessed sequentially from `start_id` to `start_id + num_pages - 1`. The `TensorAccessor` handles bank-interleaved addressing, mapping logical tile index to (bank_id, offset_within_bank) using the buffer's page size and bank configuration.

## Memory Access Patterns

### Read Pattern

Sequential tile reads. Each core reads tiles `[start_id, start_id + num_pages)` in order. Each tile read is a single `noc_async_read_page` followed by a barrier, resulting in one-tile-at-a-time sequential access to DRAM.

### Write Pattern

Sequential tile writes matching the read order. Each core writes tiles `[start_id, start_id + num_pages)` in order. Writes use `noc_async_write_page` with flush after each tile and a final barrier.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Full device compute grid (`compute_with_storage_grid_size`) |
| Work Splitting | `split_work_to_cores(grid_size, num_tiles)` |
| Core Group 1 | Cores with `ceil(num_tiles / num_cores)` tiles each |
| Core Group 2 | Cores with `floor(num_tiles / num_cores)` tiles each (remainder handling) |
| Core Enumeration | Column-major: `core = {i / num_cores_y, i % num_cores_y}` |
| Load Balancing | At most 1-tile difference between core groups |

Each core group gets its own compute kernel instance with a different `per_core_block_cnt` compile-time argument. If `core_group_2` is empty (tiles divide evenly), only one kernel instance is created.

## Arguments

### Compile-Time Arguments

**Reader kernel** (`reader_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0+ | TensorAccessorArgs | uint32_t[] | Buffer addressing parameters (bank count, page size, alignment) from `TensorAccessorArgs(*src_buffer)` |

**Writer kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output circular buffer index (= 2, i.e. `c_2`) |
| 1+ | TensorAccessorArgs | uint32_t[] | Buffer addressing parameters from `TensorAccessorArgs(*dst_buffer)` |

**Compute kernel** (`eltwise_sfpu.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for this factory) |

### Runtime Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_pages | uint32_t | Number of tiles to read |
| 2 | start_id | uint32_t | Starting tile index for this core |

**Writer kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile index for this core |

**Compute kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | packed_scalar1 | uint32_t | Always 0 for RECIP (no scalar parameter) |
| 1 | packed_scalar2 | uint32_t | Always 0 for RECIP (no scalar parameter) |

### Compute Kernel Configuration

| Property | Value |
|---|---|
| Math Fidelity | HiFi4 |
| FP32 Dest Acc | Configurable via `args.fp32_dest_acc_en` |
| Math Approx Mode | false (RECIP returns false from `get_op_approx_mode`) |
| Defines | `SFPU_OP_RECIP_INCLUDE=1`, `SFPU_OP_CHAIN_0` (init + func calls), `INP_FLOAT32` or `INP_FLOAT` |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Creates a `TensorAccessor` from compile-time args and source address. Iterates from `start_id` to `start_id + num_pages`, reading one tile per iteration into CB `c_0`. Each read is barrier-synchronized before pushing to the CB.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Creates a `TensorAccessor` from compile-time args and destination address. Iterates from `start_id` to `start_id + num_pages`, waiting for one tile in CB `c_2`, writing it to DRAM, then popping. Uses `noc_async_writes_flushed` per tile and a final `noc_async_write_barrier`.

### Compute Kernel

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // conditionally includes recip.h via SFPU_OP_RECIP_INCLUDE
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tile-blocks this core must process
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block; always 1 for standard unary

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // initializes unpack from c_0 and pack to c_2; configures HW pipelines
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for one tile
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DST register file for math thread

            // Wait for reader to produce one tile in input CB
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from c_0 position 0 to DST register 0 (unpack + move)
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // SFPU_OP_CHAIN_0 expands to:
            //   recip_tile_init<false>();
            //   recip_tile<false>(0);
            // This initializes the SFPU for reciprocal and computes 1/x on DST[0]
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            tile_regs_commit();  // signal that math thread is done writing DST; hand off to pack thread

            tile_regs_wait();  // pack thread waits for DST data to be ready

            pack_tile(0, tt::CBIndex::c_2);  // pack DST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);  // free the consumed input tile from c_0

            tile_regs_release();  // release DST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publish the produced tile(s) to writer
    }
}
```

### SFPU Kernel Implementation

The RECIP operation has significantly different SFPU implementations between Blackhole and Wormhole B0 architectures.

#### SFPU Kernel File (Blackhole)

`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`

#### Annotated SFPU Kernel Source (Blackhole)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_rsqrt_compat.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Computes the reciprocal of a floating point value x.
// Uses hardware approx_recip instruction for initial estimate, then Newton-Raphson refinement.
// max_iter=0: approx_recip only (~7-bit precision)
// max_iter=1: one Newton-Raphson iteration (~16-bit precision, sufficient for BF16)
// max_iter=2: two Newton-Raphson iterations (~24-bit precision, sufficient for FP32)
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat x)
{
    // SFPARECIP hardware instruction: returns approximate reciprocal
    // Returns +/-0 for +/-inf or values >= 2^126, and +/-inf for +/-0
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Optionally improve the approximation using Newton-Raphson.
    if constexpr (max_iter > 0)
    {
        // Newton-Raphson: normally t = 2.0 - x*y, but we compute t = x*y - 2.0
        // (negated form). This makes NaN detection simpler via sign check.
        // vConstFloatPrgm0 was initialized to 2.0f in _init_sfpu_reciprocal_.
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;  // SFPMAD: t = x*y - 2.0

        if constexpr (max_iter > 1)
        {
            // First refinement: y1 = y * (-t) - 0 = -y*t
            // Uses vConst0 (0.0) as addend in SFPMAD
            sfpi::vFloat y1 = y * -t - sfpi::vConst0;  // SFPMAD: y1 = y*(-t) + 0

            // NaN guard: if x=0 and y=inf (or vice versa), t=NaN, and NaN >= 0
            // So we only apply refinement when t < 0 (normal case)
            v_if (t < 0)  // SFPSETCC: conditional execution based on sign of t
            {
                // Second Newton-Raphson iteration on the refined y1
                t = x * y1 - sfpi::vConstFloatPrgm0;  // SFPMAD: t = x*y1 - 2.0
                y = y1 * -t - sfpi::vConst0;           // SFPMAD: y = y1*(-t)
            }
            v_endif;  // SFPENCC: end conditional
        }
        else
        {
            // Single iteration refinement
            v_if (t < 0)  // skip if NaN (t >= 0 when NaN)
            {
                y = y * -t - sfpi::vConst0;  // SFPMAD: y = y*(-t)
            }
            v_endif;
        }
    }

    return y;
}

// Approximate reciprocal macro-based path, throughput 1 cycle per 32 elements.
// Uses SFPLOADMACRO with pre-configured macro 0 which performs:
// load -> approx_recip -> store (all in one macro invocation per element batch).
inline void _calculate_reciprocal_fast_7b_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        // SFPLOADMACRO: macro 0 loads from DEST, applies SFPARECIP, stores back
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_6, 0);
    }
    TTI_SFPNOP;  // pipeline drain
    TTI_SFPNOP;
}

// BF16-precision reciprocal, throughput 3 cycles per 32 elements.
// Uses SFPLOADMACRO pipeline: load -> approx_recip -> Newton-Raphson correction -> store.
// Corrects the LSB of the BF16 result for improved accuracy.
inline void _calculate_reciprocal_fast_8b_3c_(const int iterations)
{
    // Pipeline uses L0=0x80000000 for LSB correction and L7=x index throughout.
    // Steps per element batch:
    //   y = load(); x = 0*0 + y; y = arecip(y);
    //   y = y | (1<<15);  -- via LO16_ONLY load of 0x8000
    //   y = x * y - 1;
    //   t = y >> 16;      -- via LO16 load
    //   y += t; store(y);

    constexpr int x           = p_sfpu::LREG1;
    constexpr int t           = p_sfpu::LREG1;
    constexpr int offset      = 0;
    constexpr int prev_offset = -4 & 0x3ff;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);  // L0 = 0x80000000 (BF16 constant)
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, x);       // L7 = register index for x

    // Prologue: first two iterations with SFPNOP in place of 3rd macro
    const int fill_end = iterations < 2 ? iterations : 2;
#pragma GCC unroll 2
    for (int d = 0; d < fill_end; d++)
    {
        int y = 3 + (d % 3);
        TT_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TTI_SFPNOP;
        TT_SFPLOADMACRO((1 << 2) | (y & 3), 14, ADDR_MOD_6, offset | (y >> 2));
    }

    // Main loop: all three macro stages active simultaneously (pipelined)
#pragma GCC unroll 6
    for (int d = 2; d < iterations; d++)
    {
        int y = 3 + (d % 3);
        TT_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TT_SFPLOADMACRO((2 << 2) | (t & 3), 9, ADDR_MOD_7, prev_offset | (t >> 2));
        TT_SFPLOADMACRO((1 << 2) | (y & 3), 14, ADDR_MOD_6, offset | (y >> 2));
    }

    // Fill gap with NOPs when iterations < 2
#pragma GCC unroll 2
    for (int d = iterations; d < 2; d++)
    {
        TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP;
    }

    // Epilogue: drain the pipeline
    const int drain_start = iterations < 2 ? 2 : iterations;
#pragma GCC unroll 2
    for (int d = drain_start; d < iterations + 2; d++)
    {
        TTI_SFPNOP;
        TT_SFPLOADMACRO((2 << 2) | (t & 3), 9, ADDR_MOD_6, prev_offset | (t >> 2));
        TTI_SFPNOP;
    }

    TTI_SFPNOP;
}

// FP32-precision reciprocal, throughput 5 cycles per 32 elements.
// Uses replay buffer and multi-stage Newton-Raphson with cubic error correction.
// Pseudocode: y = arecip(x); e = 1-x*y; t = e*e+e; t2 = t*e+e; t2 = min(t2,1.0); y = t2*y+y
inline void _calculate_reciprocal_fast_24b_5c_(const int iterations)
{
    constexpr int e  = p_sfpu::LREG0;
    constexpr int t2 = p_sfpu::LREG1;
    constexpr int z  = p_sfpu::LREG2;
    constexpr int y  = p_sfpu::LREG3;

    // Replay buffers contain pre-programmed instruction sequences
    lltt::replay(0, 4);               // replay instructions 0-3 (load, arecip, error calc, etc.)
    TTI_SFPLOAD(7, 0, ADDR_MOD_6, 0); // load from DEST

#pragma GCC unroll 7
    for (int d = 0; d < iterations - 1; d++)
    {
        lltt::replay(0, 5);  // replay full 5-instruction sequence for each additional iteration
    }

    TTI_SFPNOP;
    lltt::replay(1, 1);  // partial replay for pipeline drain
    TTI_SFPNOP;
    lltt::replay(3, 2);  // final drain

    TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP;  // pipeline flush
}

// Dispatch function: selects the appropriate fast path based on precision requirements.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations)
{
    if constexpr (APPROXIMATION_MODE)
    {
        _calculate_reciprocal_fast_7b_(iterations);      // ~7-bit, 1c/32 throughput
    }
    else if constexpr (is_fp32_dest_acc_en)
    {
        _calculate_reciprocal_fast_24b_5c_(iterations);  // ~24-bit, 5c/32 throughput
    }
    else
    {
        _calculate_reciprocal_fast_8b_3c_(iterations);   // ~8-bit, 3c/32 throughput
    }
}

// Top-level entry point called from recip_tile -> calculate_reciprocal.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_reciprocal_(const int iterations)
{
    if constexpr (legacy_compat)
    {
        // Legacy path uses _sfpu_reciprocal_ directly via _calculate_reciprocal_compat_
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
    else
    {
        // Optimized path using SFPLOADMACRO-based instruction sequences
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
}

// Initialization for 7-bit approximate reciprocal path.
// Configures SFPLOADMACRO macro 0 to: load -> arecip -> store with L16 format.
inline void _init_reciprocal_fast_7b_()
{
    TTI_SFPARECIP(0, 0, 12, 0);  // program the SFPARECIP instruction template

    constexpr std::uint32_t simple_bits = 0x00 | 0x40 | (0 << 3) | (4 + 0);
    constexpr std::uint32_t mad_bits    = 0;
    constexpr std::uint32_t round_bits  = 0;
    constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;

    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
    TTI_SFPCONFIG(0, 4, 0);        // configure macro 0
    TTI_SFPCONFIG(0x110, 8, 1);    // set UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1
}

// Initialization for 8-bit BF16 reciprocal path.
// Configures instruction templates and three SFPLOADMACRO macros for the 3-cycle pipeline.
inline void _init_reciprocal_fast_8b_3c_()
{
    constexpr int x = p_sfpu::LREG1;
    constexpr int t = p_sfpu::LREG1;

    TTI_SFPARECIP(0, 0, 12, 0);
    TTI_SFPMAD(p_sfpu::LCONST_0, p_sfpu::LCONST_0, 0, 13, 8); // x = 0*0 + y (copy y to x via MAD)
    TTI_SFPMAD(x, 0, p_sfpu::LCONST_neg1, 14, 0);              // e = x*y - 1
    TTI_SFPIADD(0, t, 15, sfpi::SFPIADD_MOD1_CC_NONE);         // y += t (integer add for LSB correction)

    // Configure macro 0, 1, 2 with appropriate simple/mad/store bit patterns
    // [Macro configurations set up the 3-stage pipeline for load->arecip->correct->store]
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0x00 | 0x00 | (0 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x80 | 0x00 | (0 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (5 << 3) | (4 + 3);
        constexpr std::uint32_t mad_bits    = 0x80 | 0x40 | (0 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (2 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (1 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    TTI_SFPCONFIG(0x700, 8, 1);  // set WaitForElapsedInstructions=1 for all 3 macros
}

// Initialization for 24-bit FP32 reciprocal path.
// Configures instruction templates, 4 SFPLOADMACRO macros, and a replay buffer.
inline void _init_reciprocal_fast_24b_5c_()
{
    constexpr int e  = p_sfpu::LREG0;
    constexpr int t2 = p_sfpu::LREG1;
    constexpr int z  = p_sfpu::LREG2;
    constexpr int y  = p_sfpu::LREG3;

    TTI_SFPARECIP(0, 0, 12, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, 13, 0);
    TTI_SFPMAD(t2, p_sfpu::LREG0, z, 14, 0);
    TTI_SFPSWAP(0, p_sfpu::LCONST_1, 15, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // clamp: min(t2, 1.0)

    // Configure 4 macros and load replay buffer with the 5-cycle instruction sequence
    // [Macro configs and replay buffer setup omitted for brevity -- see source above]
    // ...

    constexpr std::uint32_t prev_offset = -2 & 0x3ff;
    constexpr std::uint32_t offset      = 0;

    load_replay_buf(0, 6, [e, t2, z, y, offset, prev_offset] {
        TTI_SFPLOADMACRO((0 << 2) | (y & 3), 0, ADDR_MOD_7, offset | (y >> 2));
        TTI_SFPLOADMACRO((2 << 2) | (t2 & 3), 0, ADDR_MOD_7, prev_offset | (t2 >> 2));
        TTI_SFPLOADMACRO((1 << 2) | (e & 3), 0, ADDR_MOD_7, offset | (e >> 2));
        TTI_SFPMAD(p_sfpu::LREG0, y, p_sfpu::LCONST_1, 0, 1);  // negate VA
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_6, prev_offset | (z >> 2));
        TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, ADDR_MOD_7, prev_offset | (z >> 2));
    });
}

// Sets vConstFloatPrgm0 = 2.0f for the Newton-Raphson formula (t = x*y - 2.0).
// Only used when APPROXIMATION_MODE is false (i.e., when N-R refinement is active).
template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    if constexpr (!APPROXIMATION_MODE)
    {
        sfpi::vConstFloatPrgm0 = 2.0f;  // program constant register 0
    }
}

// Top-level init dispatcher: selects the fast init path based on precision mode.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _init_reciprocal_()
{
    if constexpr (!legacy_compat)
    {
        if constexpr (APPROXIMATION_MODE)
        {
            _init_reciprocal_fast_7b_();
        }
        else if constexpr (is_fp32_dest_acc_en)
        {
            _init_reciprocal_fast_24b_5c_();
        }
        else
        {
            _init_reciprocal_fast_8b_3c_();
        }
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Kernel File (Wormhole B0)

`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_recip.h`

#### Annotated SFPU Kernel Source (Wormhole B0)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Computes the reciprocal of a floating point value x using polynomial initial estimate
// and Newton-Raphson refinement. No hardware approx_recip available on Wormhole.
// max_iter = 2: sufficient for float32 precision (<= 1 ulps)
// max_iter = 1: sufficient for bfloat16/float16 precision (<= 0.5 ulps)
// max_iter = 0: same as max_iter=1 currently; may use cheaper approximation in future
template <int max_iter = 2>
sfpi_inline sfpi::vFloat _sfpu_reciprocal_(const sfpi::vFloat in)
{
    // Scale input to [1.0, 2.0) range and make negative.
    // setman copies the mantissa of `in` into a float with exponent/sign of -1.0.
    // If in != +/-0 and in != +/-inf, then x = in * 2^(127 - in.Exp).
    // If in = +/-0 or in = +/-inf, then x = +/-1.
    sfpi::vFloat negative_x = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(in));

    // Quadratic initial estimate: y = k2 + k1*(-x) = k2 - k1*x
    // where k0, k1, k2 are Sollya-optimized polynomial coefficients.
    // vConstFloatPrgm0 = 0.3232325... (k0), vConstFloatPrgm1 = 1.4545459... (k1)
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Compute scale factor: 1/in = 1/x * 2^(127 - in.Exp)
    // scale.Exp = 255 - in.Exp = ~in.Exp (bitwise NOT of input's raw bits)
    // This naturally handles special cases: in.Exp=0 -> scale=inf, in.Exp=255 -> scale=0
    sfpi::vInt scale_bits = ~sfpi::reinterpret<sfpi::vUInt>(in);

    // Continue quadratic estimate: y = k2 + y * (-x)
    // vConstFloatPrgm2 = 2.121212... (k2)
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Clear mantissa of scale factor to get a pure power-of-two
    sfpi::vFloat scale = sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(scale_bits), 0);

    // First Newton-Raphson iteration: t = 1.0 - x*y (using negative_x, so t = 1.0 + (-x)*y)
    sfpi::vFloat t = sfpi::vConst1 + negative_x * y;

    // Adjust scale by 0.5 to correct for the 255 vs 254 exponent bias.
    // Also handles inf*0.5=inf and 0*0.5=0 correctly for special cases.
    scale *= 0.5f;

    // Apply first N-R correction: y = y + y*t = y * (1 + t) = y * (2 - x*y_prev)
    y = y + y * t;

    if constexpr (max_iter > 1)
    {
        // Second Newton-Raphson iteration for FP32 precision
        t = sfpi::vConst1 + negative_x * y;  // t = 1.0 - x*y
        y = y + y * t;                        // y = y * (2 - x*y)
    }

    // Apply scaling: multiply by power-of-two to undo the normalization
    y = y * scale;
    // Set the sign of the result to match the input
    y = sfpi::setsgn(y, in);

    return y;
}

// _calculate_reciprocal_internal_: iterates over tile faces (8 iterations = 4 faces * 2 halves)
// Each iteration reads from dst_reg[0], computes reciprocal, writes back, advances dst_reg.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];  // read 32 elements from current DEST row

        if constexpr (APPROXIMATION_MODE)
        {
            // max_iter=0: single iteration (cheapest)
            sfpi::dst_reg[0] = _sfpu_reciprocal_<0>(in);
        }
        else
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                // max_iter=2: two N-R iterations for FP32 precision
                sfpi::dst_reg[0] = _sfpu_reciprocal_<2>(in);
            }
            else
            {
                // max_iter=1: one N-R iteration for BF16 precision
                // Convert result to fp16b format before writing back
                sfpi::vFloat out = _sfpu_reciprocal_<1>(in);
                sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(out, 0));
            }
        }

        sfpi::dst_reg++;  // advance to next row of 32 elements in DEST
    }
}

// Top-level dispatcher: legacy_compat selects between compat and optimized paths.
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _calculate_reciprocal_(const int iterations)
{
    if constexpr (legacy_compat)
    {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
    else
    {
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(iterations);
    }
}

// Initialize programmable constant registers for the polynomial initial estimate.
// These Sollya-optimized coefficients minimize the maximum relative error of 1/x
// over [1, 2) for the quadratic polynomial y = k2 - k1*x + k0*x^2.
template <bool APPROXIMATION_MODE>
inline void _init_sfpu_reciprocal_()
{
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;   // k0 coefficient
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;         // k1 coefficient
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;       // k2 coefficient
}

// Top-level init: for non-legacy mode, delegates to _init_sfpu_reciprocal_.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
inline void _init_reciprocal_()
{
    if constexpr (!legacy_compat)
    {
        _init_sfpu_reciprocal_<APPROXIMATION_MODE>();
    }
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

**Blackhole-specific instructions:**

| Instruction | Description |
|---|---|
| `SFPARECIP` (`approx_recip`) | Hardware approximate reciprocal (~7-bit precision). Blackhole-only. Returns +/-0 for +/-inf, +/-inf for +/-0. |
| `SFPMAD` | Fused multiply-add: `result = VA * VB + VC`. Core arithmetic for Newton-Raphson steps. |
| `SFPLOADMACRO` | Macro-based instruction that combines load, simple op, MAD, and store in a single pipelined invocation. Used in all three fast paths. |
| `SFPLOADI` | Load immediate value into SFPU local register or configure macro bit fields. |
| `SFPCONFIG` | Configure SFPU macro execution parameters (store format, wait behavior). |
| `SFPNOP` | No-operation; used for pipeline draining and timing. |
| `SFPIADD` | Integer add on SFPU registers; used in 8-bit path for LSB correction. |
| `SFPSWAP` | Swap/min/max operation; used in 24-bit path for `min(t2, 1.0)` clamping. |
| `SFPLOAD` | Load data from DEST register file into SFPU local register. |
| `SFPSETCC` / `SFPENCC` | Set/end conditional code for `v_if`/`v_endif` predicated execution. |

**Wormhole B0-specific instructions:**

| Instruction | Description |
|---|---|
| `setman` | Set mantissa of a float from another value; used for range normalization to [1, 2). |
| `setsgn` | Set sign bit of a float; used to restore output sign from input. |
| `reinterpret` | Bitwise reinterpret cast between vFloat/vInt/vUInt. |
| `SFPMAD` (implicit) | Fused multiply-add via `*` and `+` operators on vFloat. Core of Newton-Raphson. |
| `SFPNOT` (implicit) | Bitwise NOT via `~` operator on vUInt; used for scale factor computation. |
| `v_if` / `v_endif` | Predicated execution (no branching, uses SFPU condition codes). |

#### SFPU Register Usage

**Programmable constant registers:**

| Register | Blackhole Usage | Wormhole B0 Usage |
|---|---|---|
| `vConstFloatPrgm0` | 2.0f (Newton-Raphson constant) | 0.3232325f (k0 polynomial coefficient) |
| `vConstFloatPrgm1` | Not used | 1.4545459f (k1 polynomial coefficient) |
| `vConstFloatPrgm2` | Not used | 2.121212f (k2 polynomial coefficient) |

**Built-in constants used:**

| Constant | Value | Purpose |
|---|---|---|
| `vConst0` | 0.0f | Zero addend in SFPMAD (Blackhole) |
| `vConst1` | 1.0f | Newton-Raphson: `t = 1.0 - x*y` (Wormhole) |
| `vConstNeg1` | -1.0f | Source of sign+exponent for mantissa extraction (Wormhole) |

**Local registers (Blackhole fast paths):**

| Register | Fast-7b | Fast-8b | Fast-24b |
|---|---|---|---|
| `LREG0` | -- | 0x80000000 constant | error `e` |
| `LREG1` | -- | `x` / `t` (shared) | `t2` |
| `LREG2` | -- | -- | `z` |
| `LREG3` | -- | `y` (rotating 3,4,5) | `y` |
| `LREG7` | -- | register index for x | -- |

**DEST registers:** `dst_reg[0]` through `dst_reg[7]` (8 iterations for a full tile = 4 faces x 2 halves x 32 lanes).

#### SFPU Execution Flow

1. **Initialization** (`recip_tile_init<false>()` called once):
   - Calls `recip_init<APPROX=false, DST_ACCUM_MODE, legacy_compat=false>()`.
   - Calls `_init_reciprocal_<false, DST_ACCUM_MODE, false>()`.
   - **Blackhole**: Selects `_init_reciprocal_fast_8b_3c_()` (or `_fast_24b_5c_()` if FP32 dest acc). Configures SFPLOADMACRO instruction templates and macro parameters.
   - **Wormhole B0**: Calls `_init_sfpu_reciprocal_<false>()` which programs the three polynomial coefficients into `vConstFloatPrgm0/1/2`.

2. **Per-tile execution** (`recip_tile<false>(0)` called per tile):
   - Expands via `SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN` to `_llk_math_eltwise_unary_sfpu_params_<false>(ckernel::sfpu::calculate_reciprocal<false, DST_ACCUM_MODE, 8, false>, 0, VectorMode::RC)`.
   - The LLK params function sets up the DEST register pointer and calls `calculate_reciprocal` which calls `_calculate_reciprocal_`.
   - Iterates 8 times (one per 32-element row in the tile's 4 faces x 2 halves = 1024 elements total):
     - **Blackhole**: Executes `SFPLOADMACRO`-based pipelined sequences that load from DEST, apply `SFPARECIP` + Newton-Raphson correction, and store back.
     - **Wormhole B0**: Reads `dst_reg[0]`, computes `_sfpu_reciprocal_<1>` (BF16) or `_sfpu_reciprocal_<2>` (FP32), writes back to `dst_reg[0]`, advances `dst_reg++`.

3. **Result propagation**: After the SFPU kernel completes, the result is in DEST registers. The compute kernel calls `tile_regs_commit()` to hand off to the pack thread, which calls `pack_tile(0, c_2)` to write from DEST to the output circular buffer.

#### SFPU Configuration

| Configuration | Value | Description |
|---|---|---|
| `APPROXIMATION_MODE` | `false` | RECIP always uses non-approximate mode (`get_op_approx_mode` returns false for all ops by default) |
| `legacy_compat` | `false` | Set explicitly in `recip_tile_init<false>()` and `recip_tile<false>()` |
| `ITERATIONS` | 8 | Hardcoded; processes all 8 rows of 32 elements in a tile (32x32 = 1024 total) |
| `Math Fidelity` | HiFi4 | Set in `ComputeConfig` |
| `fp32_dest_acc_en` | Runtime configurable | Determines whether 24-bit (BH) or 2-iteration N-R (WH) path is used |

#### Hardware Compatibility Notes

- **Blackhole** has the `SFPARECIP` hardware instruction providing a fast initial approximation (~7-bit). This enables three optimized paths with different precision/throughput tradeoffs:
  - Fast 7b: 1 cycle/32 elements (approx only)
  - Fast 8b: 3 cycles/32 elements (BF16 precision with LSB correction)
  - Fast 24b: 5 cycles/32 elements (FP32 precision with cubic error correction)
- **Wormhole B0** lacks `SFPARECIP` and uses a software polynomial initial estimate (`y = k2 - k1*x + k0*x^2`) followed by Newton-Raphson. This requires more SFPU cycles but achieves equivalent precision. The polynomial coefficients are Sollya-optimized to minimize maximum relative error over [1, 2).
- Both architectures normalize the input to [1, 2) range before computing the reciprocal, but via different mechanisms: Blackhole relies on `approx_recip` to handle normalization implicitly, while Wormhole B0 uses explicit `setman`/`setsgn`/bitwise-NOT for range scaling and sign handling.
- The `legacy_compat=false` template parameter (hardcoded for RECIP) selects the newer optimized paths on both architectures.

## Implementation Notes

1. **RECIP has no scalar parameters**: Unlike operations like HARDSHRINK or WHERE_TSS, RECIP does not use `packed_scalar1` or `packed_scalar2` runtime arguments. They are always 0.

2. **Shared program factory**: RECIP shares `eltwise_sfpu.cpp` compute kernel and the full `UnaryProgramFactory` with most other unary SFPU operations. The operation-specific behavior is entirely controlled through preprocessor defines (`SFPU_OP_RECIP_INCLUDE`, `SFPU_OP_CHAIN_0`).

3. **Define chain mechanism**: The `get_block_defines` function generates defines like:
   - `SFPU_OP_CHAIN_0` = `"SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0"`
   - `SFPU_OP_CHAIN_0_INIT_0` = `"recip_tile_init<false>();"`
   - `SFPU_OP_CHAIN_0_FUNC_0` = `"recip_tile<false>(0);"`

4. **BITCAST special case**: The input CB data format is overridden to use the output format when the operation is BITCAST, to avoid unpacker conversion. This does not apply to RECIP.

5. **Precision-aware dispatch**: The Blackhole implementation is notably different from Wormhole B0, using `SFPLOADMACRO`-based pipelined instruction sequences that are significantly more throughput-efficient than the Wormhole B0 software-only approach.

6. **NaN handling (Blackhole)**: The negated Newton-Raphson formula (`t = x*y - 2.0` instead of `t = 2.0 - x*y`) enables simple NaN detection via sign check (`t < 0`), since NaN always satisfies `t >= 0`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the unary program factory work for SFPU operations like RECIP?"
   **Reason**: Needed to understand the overall factory structure, kernel selection, and CB configuration.
   **Key Findings**: Confirmed reader/compute/writer kernel paths, CB indices (c_0, c_2), double-buffering with 2 pages, and `split_work_to_cores` for work distribution.

2. **Query**: "Where is the SFPU recip kernel implementation located in the codebase?"
   **Reason**: Needed to locate the SFPU kernel source files for both architectures.
   **Key Findings**: Located `ckernel_sfpu_recip.h` in both `tt_llk_blackhole` and `tt_llk_wormhole_b0` submodule paths. Identified the `_sfpu_reciprocal_`, `_calculate_reciprocal_`, and `_init_reciprocal_` function hierarchy.

3. **Query**: "Show me the complete implementation of _sfpu_reciprocal_ and related functions" (tt-llk)
   **Reason**: Needed detailed implementation of the LLK-level SFPU functions.
   **Key Findings**: Blackhole uses hardware `approx_recip` + Newton-Raphson; Wormhole B0 uses polynomial estimate + Newton-Raphson. Three precision tiers on Blackhole (7b, 8b, 24b) using SFPLOADMACRO.

4. **Query**: "How does the SFPI _sfpu_reciprocal_ intrinsic work?" (sfpi)
   **Reason**: Needed to understand the SFPI-level instruction mapping and register usage.
   **Key Findings**: Detailed Wormhole B0 algorithm: setman for normalization, polynomial estimate, Newton-Raphson, exponent adjustment via bitwise NOT, sign restoration via setsgn.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace how RECIP maps to kernel defines and compute kernel path.
   **Key Information**: RECIP -> `SFPU_OP_RECIP_INCLUDE`, compute kernel = `eltwise_sfpu.cpp` (default), init = `recip_tile_init<false>()`, func = `recip_tile<false>(0)`, approx_mode = false.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/recip.h`
   **Reason**: Needed to understand the `recip_tile` and `recip_tile_init` API layer.
   **Key Information**: `recip_tile_init` uses `SFPU_THREE_TEMPLATE_PARAM_INIT(reciprocal, sfpu::recip_init, APPROX, DST_ACCUM_MODE, legacy_compat)`. `recip_tile` uses `SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN(calculate_reciprocal, APPROX, DST_ACCUM_MODE, 8, legacy_compat, idst, vector_mode)`.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to understand the SFPU macro dispatch mechanism.
   **Key Information**: `SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN` expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::FN<APPROXIMATE, FP32, ITER, LEGACY_COMPAT>, DST_IDX, MODE)`.
