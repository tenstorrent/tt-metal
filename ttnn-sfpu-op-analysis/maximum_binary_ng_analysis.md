# MAXIMUM (binary_ng) Implementation Analysis

## Overview

The MAXIMUM operation computes the element-wise maximum of two input tensors: `c[i] = max(a[i], b[i])`. It is an SFPU-only binary operation implemented through the `binary_ng` program factory framework. The operation supports float (bfloat16/float16_b), float32, int32, and uint32 data types, with dedicated SFPU kernel variants for each. It supports full N-dimensional broadcasting between input tensors.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Work Unit Definition

One work unit is a single 32x32 tile. The compute kernel processes one tile per read-compute-write cycle (`num_tiles_per_cycle = 1`). Each cycle loads one tile from each input (LHS, RHS), performs the SFPU max operation, and writes one output tile.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Input A (LHS) | Input B (RHS) |
|---|---|---|
| Dimension Convention | [..., D, N, C, H, W] (up to N-D, dims >5 collapsed) |  [..., D, N, C, H, W] (broadcastable against A) |
| Tensor Layout | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded (Height/Width/Block) | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | bfloat16, float32, int32, uint32 | Same as A (must match) |

### Output Tensor(s)

| Property | Output C |
|---|---|
| Dimension Convention | Broadcast-expanded shape of A and B |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | Same as input dtype |

### Layout Transformations

No tilize/untilize is performed. Inputs and outputs must already be in tiled layout. When sharding is used, the program factory computes per-core shard shapes and adjusts tile offsets accordingly. If native L1 sharding conditions are not met (e.g., uneven shards, different grids), the operation falls back to interleaved tensor accessor mode.

## Data Flow Pattern

1. **Reader kernel** reads one tile from input A into CB c_0 and one tile from input B into CB c_1 (per iteration). For sharded inputs, the reader simply reserves and pushes the entire shard at once.
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1, acquires DST registers, copies LHS tile to DEST[0] and RHS tile to DEST[1] using `copy_tile`, then invokes `BINARY_SFPU_OP(0, 1, 0)` which dispatches `binary_max_tile(0, 1, 0)`. The result overwrites DEST[0]. The result is packed into CB c_2.
3. **Writer kernel** waits for a tile in CB c_2 and writes it to the output buffer via NoC.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Data Format | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src_a / cb_pre_lhs | Input tensor A | a_data_format | 2 (interleaved) or shard volume | 1 | Double (interleaved) / Single (sharded) | Reader | Compute |
| c_1 | cb_src_b / cb_pre_rhs | Input tensor B | b_data_format | 2 (interleaved) or shard volume | 1 | Double (interleaved) / Single (sharded) | Reader | Compute |
| c_2 | cb_out | Output tensor C | c_data_format | 2 (interleaved) or shard volume | 1 | Double (interleaved) / Single (sharded) | Compute | Writer |
| c_3 | cb_post_lhs | LHS after pre-activation | a_data_format | 1 | 1 | Single | Compute (preprocess) | Compute (main) |
| c_4 | cb_post_rhs | RHS after pre-activation | b_data_format | 1 | 1 | Single | Compute (preprocess) | Compute (main) |

For MAXIMUM, there are no LHS/RHS pre-activations (`process_lhs` and `process_rhs` are `std::nullopt`), so CB c_3 and c_4 are not created. The compute kernel uses `cb_post_lhs = cb_pre_lhs = c_0` and `cb_post_rhs = cb_pre_rhs = c_1` directly.

Similarly, CB c_5 and c_6 (used for row broadcast) are not created for the NONE broadcast case.

## Pipeline Pattern Summary

- **Interleaved mode**: CB c_0 and c_1 have capacity=2, enabling double-buffering between reader and compute. CB c_2 has capacity=2, enabling double-buffering between compute and writer.
- **Sharded mode**: CBs are sized to the full shard volume, functioning as single-buffered storage since the reader pushes all tiles at once.

## Index Calculations

The reader kernel computes a multi-dimensional tile offset from the flat `start_tile_id` by decomposing it into (nd, d, n, c, th, tw) coordinates. For each input tensor, stride values are computed that account for broadcasting: when a dimension has size 1 in the input but >1 in the output, the stride for that dimension is 0 (via the `(dim > 1)` multiplier pattern in the program factory).

The output tile offset is computed as:
```
tile_offset = start_nd * nD_stride + start_d * d_stride + start_n * n_stride + start_c * c_stride + start_th * Wt + tw
```

For sharded mode, the start tile ID for each core is calculated from core position:
```
c_start_id = (core_index / num_shards_per_width) * (c_shard_height * cWt) + (core_index % num_shards_per_width) * c_shard_width
```

## Memory Access Patterns

### Read Pattern

For interleaved mode, tiles are read one at a time from both A and B in the nested loop order: nD -> D -> N -> C -> H -> W (innermost). Both inputs are read with the same outer loop structure but potentially different stride values (to handle broadcasting). Each tile read is followed by a `noc_async_read_barrier()`.

For sharded mode, data is already in L1, so the reader just performs `cb_reserve_back`/`cb_push_back` to make tiles available.

### Write Pattern

The writer iterates with the same nested dimension order as the reader. Each output tile is written via `noc_async_write_page` followed by a barrier. For sharded output, the writer section is compiled out entirely.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Worker grid from `operation_attributes.worker_grid` |
| Work Splitting | `split_work_to_cores` divides total output tiles across cores |
| Core Group 1 | Cores with `num_tiles_per_core_group_1` tiles (ceiling division) |
| Core Group 2 | Remaining cores with `num_tiles_per_core_group_2` tiles (one fewer) |
| Load Balancing | Difference of at most 1 tile between groups |
| Remainder Handling | Cores outside both groups get zero-filled runtime args (NOOP) |
| Sharded Mode | Grid comes from shard spec; each core processes its own shard |

When `zero_start_grid` is true (single rectangular grid starting at (0,0)), an optimized path is used. Otherwise, a generic `CoreRangeSet`-based distribution handles arbitrary grids.

## Arguments

### Compile-Time Arguments

**Compute Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1. Number of output tiles produced per compute cycle |

The compute kernel also receives preprocessor defines:
- `BINARY_SFPU_INIT` = `binary_max_tile_init();` (or int32/uint32 variant)
- `BINARY_SFPU_OP` = `binary_max_tile` (or `binary_max_int32_tile` / `binary_max_uint32_tile`)
- `BCAST_INPUT` = `""` (no broadcast for NONE mode)
- `PROCESS_LHS_ACTIVATIONS(i)` = empty (no pre-activation)
- `PROCESS_RHS_ACTIVATIONS(i)` = empty
- `PROCESS_POST_ACTIVATIONS(i)` = empty (no post-activation)

**Reader Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0..K | TensorAccessorArgs for A | uint32_t[] | Compile-time accessor config for input A |
| K+1..M | TensorAccessorArgs for B | uint32_t[] | Compile-time accessor config for input B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding is active |

**Writer Kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0..K | TensorAccessorArgs for C | uint32_t[] | Compile-time accessor config for output C |
| K+1 | has_sharding | uint32_t | 1 if native L1 sharding is active |

### Runtime Arguments

**Compute Kernel (per-core):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | c_num_tiles | uint32_t | Number of output tiles this core must process |
| 1 | freq | uint32_t | Broadcast frequency (1 for NONE mode) |
| 2 | counter | uint32_t | Broadcast counter start (0 for NONE mode) |
| 3 | compute_scalar_value | uint32_t | Unused for MAXIMUM (set to 0) |

**Reader Kernel (per-core):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | DRAM/L1 address of input A buffer |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core |
| 2 | src_num_tiles | uint32_t | Number of A tiles (sharded) or 0 |
| 3 | dst_num_tiles | uint32_t | Number of output tiles this core processes |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (sharded) or 0 |
| 5 | nD_stride | uint32_t | A's stride for collapsed dims >5 |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9-14 | D, N, C, Ht, Wt, cND | uint32_t | Output shape dimensions |
| 15 | src_addr_b | uint32_t | DRAM/L1 address of input B buffer |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed dims >5 |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles (sharded) or 0 |

**Writer Kernel (per-core, tensor-tensor mode):**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output C buffer |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core |
| 2 | dst_num_tiles | uint32_t | Number of tiles to write |
| 3 | dst_shard_width | uint32_t | Shard width (sharded) or 0 |
| 4-10 | D, N, C, Ht, Wt, cND, 0 | uint32_t | Output shape + padding |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` (for NONE broadcast mode)
- **Key Logic**: Reads both A and B tiles in a nested 6D loop (nD, D, N, C, Ht, Wt). Uses stride-based offsets to handle broadcasting: when an input dimension is 1 but the output dimension is >1, the corresponding stride is 0, causing repeated reads of the same tile. For sharded inputs, the kernel simply does `cb_reserve_back`/`cb_push_back` to expose the pre-existing L1 data to the compute kernel.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Writes output tiles using the same nested 6D loop. Uses `TensorAccessor` for address computation. For sharded outputs, the entire writer body is compiled out.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
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
#include "api/compute/quantization.h"
#include "api/compute/binary_max_min.h"          // provides binary_max_tile, binary_max_tile_init, etc.
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"               // macro infrastructure: HAS_ACTIVATIONS, BCAST_OP, etc.
#include "eltwise_utils_sfpu.hpp"                  // PREPROCESS macro for pre-activations

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0); // runtime arg: total tiles this core must process

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0); // always 1 for this kernel

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;  // input A circular buffer
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;  // input B circular buffer
    constexpr auto cb_out = tt::CBIndex::c_2;       // output circular buffer

    // For MAXIMUM: HAS_ACTIVATIONS(LHS)=0, HAS_ACTIVATIONS(RHS)=0, so cb_post_lhs=cb_pre_lhs=c_0
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);     // initializes unpack/pack configuration
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU))); // not used for MAXIMUM
#endif

    // For MAXIMUM with no activations: BINARY_SFPU_INIT is called once here
    // Expands to: binary_max_tile_init();
    // This configures the SFPU pipeline, sets up SFPLOADMACRO instruction templates and macros
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS(LHS, ...) is a no-op for MAXIMUM (no LHS activations)
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle); // wait for reader to produce 1 LHS tile in c_0

        // PREPROCESS(RHS, ...) is a no-op for MAXIMUM (no RHS activations)
        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle); // wait for reader to produce 1 RHS tile in c_1

        cb_reserve_back(cb_out, num_tiles_per_cycle);    // reserve space for 1 output tile in c_2

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT  // re-init after activation changed SFPU state (not taken for MAXIMUM)
#endif
        tile_regs_acquire();  // acquire DST register file for math operations

        // Configure unpack for RHS->DST copy, then copy LHS tiles
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);     // unpack LHS tile 0 from c_0 into DEST[0]
        }

        // Reconfigure unpack for LHS->DST, then copy RHS tiles and perform SFPU op
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1); // unpack RHS tile 0 from c_1 into DEST[1]

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT  // not taken for MAXIMUM
#endif
            // Core SFPU operation: binary_max_tile(0, 1, 0)
            // Reads DEST[0] (LHS) and DEST[1] (RHS), computes max, writes result to DEST[0]
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // PROCESS_POST_ACTIVATIONS(0) is empty for MAXIMUM (no postprocess)
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();   // signal that DST registers are ready for packing

        tile_regs_wait();     // wait for commit to complete

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);  // pack DEST[0] result into output CB c_2
        }
        tile_regs_release();  // release DST register file

        cb_push_back(cb_out, num_tiles_per_cycle);   // notify writer that 1 tile is ready
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle); // free consumed LHS tile
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle); // free consumed RHS tile
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`
(identical structure for `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`)

#### Annotated SFPU Kernel Source

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

// Floating-point max/min: uses SFPSWAP with VEC_MIN_MAX mode
// IS_MAX_OP=true selects maximum, false selects minimum
// ITERATIONS=8 processes all 8 rows of a tile face (each row is 32 elements)
template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Compute byte offsets into DEST register. Each tile position occupies 32 rows,
    // and each row is 2 bytes wide in the SFPU view, hence the <<1 shift.
    uint offset0 = (dst_index_in0 * 32) << 1;  // DEST offset for input 0 (LHS)
    uint offset1 = (dst_index_in1 * 32) << 1;  // DEST offset for input 1 (RHS)
    uint offset2 = (dst_index_out * 32) << 1;   // DEST offset for output

#ifdef DISABLE_SFPLOADMACRO
    // Fallback path: simple but slower (4 cycles per row instead of 3)
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load two values from DEST into SFPU local registers
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0);  // LREG0 = LHS row
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);  // LREG1 = RHS row
        // SFPSWAP with VEC_MIN_MAX mode: after swap, LREG1 = max(LREG0,LREG1), LREG0 = min
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        // Store maximum (LREG1) or minimum (LREG0) depending on IS_MAX_OP
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2);
    }
#else
    // Optimized path using SFPLOADMACRO: achieves 3 cycles per row throughput.
    // SFPLOADMACRO pipelines load/simple/round/store stages across iterations.
    //
    // Pipeline schedule per row:
    // t=0: Load [a] from input0            (via SFPLOADMACRO macro 0)
    // t=1: Load b from input1              (standard SFPLOAD)
    // t=2: Load [c] + swap_minmax([a], b)  (via SFPLOADMACRO macro 1, simple stage runs SFPSWAP)
    // ... pipelined: round stage converts [a] result, store stage writes [c] to output

    constexpr int b = p_sfpu::LREG2;   // scratch register for RHS load
    constexpr int c = p_sfpu::LREG3;   // scratch register for output store

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate LREG0/LREG1 to avoid pipeline stalls
        // SFPLOADMACRO macro 0: loads input0 row into register 'a', triggers simple-stage SFPSWAP
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0 | (a >> 2));
        // Standard load of input1 row into LREG2
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        // SFPLOADMACRO macro 1: triggers round (L16 conversion) and store to output
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2 | (c >> 2));
    }

    // Pipeline drain: 3 NOPs to flush the pipeline after the last iteration
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

// Integer max/min: uses SFPSWAP + SFPSETCC for correct signed/unsigned handling
// IS_MAX_OP=true selects maximum, IS_UNSIGNED=true for uint32 comparison
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Load as INT32 format
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, offset1);
        // SFPSWAP with min/max mode; for unsigned, mod1=9 inverts the comparison sense
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // SFPSWAP operates on the float representation, so for integers the result
        // can be incorrect when signs differ. Fix with conditional swap:
        // Set condition code based on sign of each register
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        // Conditionally swap to correct ordering for elements where SFPSWAP was wrong
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        TTI_SFPENCC(0, 0, 0, 0);  // clear condition codes

        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, offset2);
    }
#else
    // Optimized SFPLOADMACRO path for int32: 5 cycles per row throughput
    // Uses replay buffer for the 10-instruction sequence (2 rows per replay)

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;

    // Record 10 instructions into replay buffer starting at slot 0
    // (Blackhole uses load_replay_buf; Wormhole uses lltt::record<lltt::NoExec>)
    load_replay_buf(0, 10, [offset0, offset1, offset2] {
        // First iteration pair: process a0,b0
        TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a0 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b0 >> 2));
        TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));

        // Second iteration pair: process a1,b1
        TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset0 | (a1 >> 2));
        TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, offset1 | (b1 >> 2));
        TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_6, offset2 | (c >> 2));
    });

    // Replay the 10-instruction block ITERATIONS/2 times (4 replays for 8 iterations)
#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);
    }

    // Handle odd iteration count and drain pipeline
    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);  // drain: replay SETCC+ENCC to finish last pair
    }

    TTI_SFPNOP;  // final pipeline drain
#endif
}

// Initialization for floating-point max/min: configures SFPLOADMACRO templates and macros
template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]: SFPSWAP that will be triggered by SFPLOADMACRO's simple stage
    // mod1=9 for IS_MAX_OP=true: VD gets max, VC gets min
    // SFPSWAP_MOD1_VEC_MIN_MAX for IS_MAX_OP=false: VD gets min, VC gets max
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[1]: SFPSHFT2 used as a no-op/identity for round stage
    TTI_SFPSHFT2(0, 0, 13, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0 configuration: triggers template[0] (SFPSWAP) in simple stage,
    // and template[1] (SFPSHFT2 identity) in round stage for format conversion
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);  // install as Macro 0
    }

    // Macro 1 configuration: store stage writes result to DEST
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);  // install as Macro 1
    }

    // Configure misc settings: StoreMod0=DEFAULT, delay kind for pipeline timing
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}

// Initialization for int32/uint32 max/min: more complex due to conditional swap correction
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int b1 = p_sfpu::LREG3;

    // Template[0]: SFPSWAP for first pair (a0, b0)
    TTI_SFPSWAP(0, b0, 12, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    // Template[1]: SFPSWAP for second pair (a1, b1)
    TTI_SFPSWAP(0, b1, 13, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    // Template[2]: SFPSETCC for sign-based conditional correction
    TTI_SFPSETCC(0, 0, 14, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    // Template[3]: SFPSHFT2 identity for round stage
    TTI_SFPSHFT2(0, 0, 15, 6);

    // Macro 0: load a0, trigger swap in simple stage, setcc+round in later stages
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: load b0, similar pipeline
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // Macro 2: SETCC correction stage
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (4 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    // Macro 3: store result
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    TTI_SFPCONFIG(0xff0, 8, 1);  // misc config for all 4 macros
#endif
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|---|---|
| `SFPLOAD` | Loads a row of data from DEST register into an SFPU local register (LREG). Supports DEFAULT (float) and INT32 load modes. |
| `SFPSTORE` | Stores a row from an SFPU local register back to DEST. Supports DEFAULT and INT32 modes. |
| `SFPSWAP` | Core comparison instruction. In `VEC_MIN_MAX` mode (mod1=8): places minimum in VC and maximum in VD. In mode `9`: inverts so VD=max, VC=min. In `SWAP` mode: conditionally swaps based on condition codes. |
| `SFPSETCC` | Sets condition codes based on register values. `LREG_LT0` sets CC for elements < 0; `LREG_GTE0` sets CC for elements >= 0. Used for integer sign correction. |
| `SFPENCC` | Clears (ends) condition code state, returning to unconditional execution. |
| `SFPLOADMACRO` | Advanced pipelined instruction that combines load with scheduled execution of pre-configured instruction templates across pipeline stages (simple, MAD, round, store). |
| `SFPCONFIG` | Configures SFPLOADMACRO macros and misc settings (store mode, delay kind). |
| `SFPLOADI` | Loads immediate values into SFPU config registers, used to configure SFPLOADMACRO macro bit fields. |
| `SFPSHFT2` | Shift instruction used here in identity mode for pipeline round-stage scheduling. |
| `SFPNOP` | No-operation; used for pipeline drainage after the last iteration. |
| `load_replay_buf` / `lltt::replay` | Records and replays instruction sequences from the replay buffer (Blackhole: `load_replay_buf`; Wormhole: `lltt::record`/`lltt::replay`). |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `LREG0` | Input 0 (LHS) row data. In SFPLOADMACRO path, alternates with LREG1 across iterations. |
| `LREG1` | Input 1 (RHS) row data (simple path). In SFPLOADMACRO float path, alternates with LREG0. |
| `LREG2` | RHS row in SFPLOADMACRO float path (`b`). In int32 path, second-pair input (`a1`). |
| `LREG3` | Output staging in SFPLOADMACRO float path (`c`). In int32 path, second-pair RHS (`b1`). |
| `LREG7` | Output staging in int32 SFPLOADMACRO path (`c`). |
| DEST[idst0*32..] | Source tile face rows for input 0 |
| DEST[idst1*32..] | Source tile face rows for input 1 |
| DEST[odst*32..] | Destination for output (overwrites input 0 position when odst=idst0) |

#### SFPU Execution Flow

1. **Initialization** (`binary_max_tile_init` -> `binary_max_min_init<true>`):
   - Configures two SFPLOADMACRO instruction templates: template[0] is an SFPSWAP in VEC_MIN_MAX mode (mod1=9 for max), template[1] is an identity SFPSHFT2 for the round stage.
   - Configures two SFPLOADMACRO macros: Macro 0 pipelines load+swap+round; Macro 1 pipelines store.
   - Sets misc configuration for pipeline timing.

2. **Per-tile execution** (`binary_max_tile` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_binary_max_min<true>`):
   - The LLK params function iterates over all 4 faces of the tile (RC vector mode), calling `calculate_binary_max_min` once per face, advancing DEST by 8 rows between calls via `SETRWC`.
   - Each face call processes 8 rows (ITERATIONS=8).

3. **Per-row computation** (within `calculate_binary_max_min`):
   - **SFPLOADMACRO path (default)**: For each row, three instructions execute:
     - `SFPLOADMACRO` macro 0: loads LHS row from DEST into alternating LREG0/LREG1, and in the simple stage triggers SFPSWAP on the previously loaded pair.
     - `SFPLOAD`: loads RHS row from DEST into LREG2.
     - `SFPLOADMACRO` macro 1: in the store stage, writes the max result back to DEST.
   - The pipeline achieves 3 cycles/row throughput by overlapping load/compute/store.
   - After 8 iterations, 3 NOPs drain the pipeline.

4. **Integer path** (`calculate_binary_max_min_int32`):
   - More complex because SFPSWAP's VEC_MIN_MAX mode operates on IEEE 754 float representation, which does not correctly order signed integers when signs differ.
   - After SFPSWAP, a conditional correction is applied: SFPSETCC checks the sign of each register, then a conditional SFPSWAP fixes misordered elements.
   - Uses replay buffer (10 instructions per 2-row pair) for efficiency, achieving 5 cycles/row.

#### SFPU Configuration

- **Unpack-to-DEST mode**: For non-POWER SFPU ops, `UnpackToDestMode::UnpackToDestFp32` is used for all input CBs. This unpacks input data directly to DEST registers in FP32 format, bypassing the SRC registers.
- **FP32 DEST accumulation**: Enabled when output or both inputs are Float32, Int32, or UInt32.
- **APPROX template parameter**: Passed through but not used by max/min (no approximation needed -- the operation is exact).
- **VectorMode::RC**: The default, processing all 4 faces of a 32x32 tile (each face is 16x16).
- **SFPLOADMACRO**: When not disabled via `DISABLE_SFPLOADMACRO`, the optimized pipelined path is used. This is the default for both Wormhole and Blackhole.

#### Hardware Compatibility Notes

- The Wormhole B0 and Blackhole implementations are structurally identical, sharing the same `ckernel_sfpu_binary_max_min.h` source. The only differences are in address modifier constants (`ADDR_MOD_7`/`ADDR_MOD_6` on Blackhole vs `ADDR_MOD_3`/`ADDR_MOD_2` on Wormhole) and the replay buffer API (`load_replay_buf` lambda on Blackhole vs `lltt::record<lltt::NoExec>` followed by explicit instructions on Wormhole).
- SFPSWAP's VEC_MIN_MAX mode operates on the IEEE 754 float bit representation. This works correctly for floats but requires the sign-correction path for integers, which adds 2 extra cycles per row in the int32 variant.
- Both architectures support the SFPLOADMACRO optimization that pipelines the load/swap/store stages.

## Implementation Notes

1. **No pre/post activations for MAXIMUM**: Unlike operations like LOGADDEXP (which uses EXP as pre-activation and LOG as post-activation), MAXIMUM has no `process_lhs`, `process_rhs`, or `postprocess` set in its `OpConfig`. This means all the activation machinery (CB c_3, c_4, PREPROCESS macros) compiles away to nothing.

2. **Three dtype variants**: The operation dispatches to different SFPU functions based on input dtype:
   - `binary_max_tile` for bfloat16/float16_b/float32 (uses float SFPSWAP)
   - `binary_max_int32_tile` for int32 (adds sign correction)
   - `binary_max_uint32_tile` for uint32 (uses unsigned comparison correction)

3. **SFPLOADMACRO optimization**: The default (non-`DISABLE_SFPLOADMACRO`) path achieves 3 cycles per row for float and 5 cycles per row for int32, compared to 4 and ~8 cycles respectively in the fallback path. This is a significant throughput improvement.

4. **Broadcast handling**: While MAXIMUM itself is a pure elementwise op, the binary_ng framework handles broadcasting transparently. The `SubtileBroadcastType` determines which reader/compute kernel variants are used. For NONE (same shape), it uses the no-broadcast variants. For other broadcast modes (SCALAR, ROW, COL), different reader/compute kernels handle tile replication.

5. **SFPU-only operation**: MAXIMUM cannot be implemented on the FPU (matrix unit) since it is not one of the three FPU binary operations (ADD, SUB, MUL). Attempting to create an FPU OpConfig for MAXIMUM will throw a `TT_THROW`.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What are the different subtile broadcast modes and how do they affect kernel selection?"
   **Reason**: Needed to understand the overall framework before diving into MAXIMUM-specific details.
   **Key Findings**: Documented the SubtileBroadcastType enum, kernel selection logic, and CB configuration patterns.

2. **Query**: "How does OpConfig work for binary_ng SFPU operations? Specifically, how is the MAXIMUM binary op type configured?"
   **Reason**: Needed to trace from BinaryOpType::MAXIMUM to the actual SFPU function names.
   **Key Findings**: MAXIMUM maps to `SfpuBinaryOp::MAXIMUM`, which produces defines `binary_max_tile_init()` and `binary_max_tile` (with int32/uint32 variants).

3. **Query**: "What does _llk_math_eltwise_binary_sfpu_params_ do in tt-llk?"
   **Reason**: Needed to understand how the SFPU function is dispatched across tile faces.
   **Key Findings**: It iterates 4 faces in VectorMode::RC, calling the SFPU function per face with SETRWC advancing DEST by 8 rows between faces.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp`
   **Reason**: Contains OpConfig constructor and get_sfpu_init_fn mapping for MAXIMUM
   **Key Information**: MAXIMUM maps to `binary_max_tile` / `binary_max_int32_tile` / `binary_max_uint32_tile` based on dtype

2. **Source**: `tt_metal/hw/inc/api/compute/binary_max_min.h`
   **Reason**: Top-level compute API header for max/min operations
   **Key Information**: Documents the ALWI wrapper functions and their parameters (idst0, idst1, odst)

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h`
   **Reason**: Core SFPU kernel implementation
   **Key Information**: Contains both float and int32 SFPSWAP-based max/min implementations with SFPLOADMACRO optimization

4. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h`
   **Reason**: LLK glue layer connecting compute API to SFPU kernel
   **Key Information**: Routes through `_llk_math_eltwise_binary_sfpu_params_` with the calculate function and SfpuType enum
