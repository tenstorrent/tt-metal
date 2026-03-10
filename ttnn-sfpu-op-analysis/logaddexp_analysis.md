# LOGADDEXP Implementation Analysis

## Overview

The LOGADDEXP operation computes `log(exp(a) + exp(b))` element-wise for two input tensors. It is implemented as a **composite** binary_ng SFPU operation that decomposes into three stages: EXP preprocessing on both inputs, SFPU ADD as the core binary operation, and LOG postprocessing on the result.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

The operation is registered as `BinaryOpType::LOGADDEXP` in the binary_ng framework. The `OpConfig` constructor (in `binary_ng_utils.cpp`, line 226) configures it as:
- `process_lhs` = `UnaryOpType::EXP`
- `process_rhs` = `UnaryOpType::EXP`
- `binary_op` = `SfpuBinaryOp::ADD` (when SFPU path) or `FpuBinaryOp::ADD` (when FPU path)
- `postprocess` = `UnaryOpType::LOG`

Since LOGADDEXP uses both LHS/RHS activations AND a post activation, the operation uses all five circular buffers (c_0 through c_4) plus the output buffer (c_2).

## Work Unit Definition

One work unit is **one 32x32 tile**. The operation processes `num_tiles_per_cycle = 1` output tile per read-compute-write cycle. The total number of tiles to process per core is determined by `split_work_to_cores` which divides the output tensor's total tile count across available cores.

## Tensor Format and Layout

### Input Tensors

| Property | Input A | Input B |
|---|---|---|
| Dimension Convention | Up to 5D (higher dims collapsed to nD) | Up to 5D (higher dims collapsed to nD) |
| Tensor Layout | TILE (32x32) | TILE (32x32) |
| Memory Layout | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | FLOAT32 / BFLOAT16 | FLOAT32 / BFLOAT16 |

### Output Tensor

| Property | Output C |
|---|---|
| Dimension Convention | Up to 5D (broadcast-expanded shape) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | FLOAT32 / BFLOAT16 |

### Layout Transformations

- No tilize/untilize conversions; all tensors must already be in tiled layout.
- Broadcasting is supported across all dimensions via the `SubtileBroadcastType` mechanism. The reader kernel handles broadcast by computing per-dimension stride masks that zero out when a dimension is size 1.
- For sharded tensors, the program may fall back to interleaved (tensor accessor) mode if native L1 sharding conditions are not met (uneven shards, different grids, DRAM buffers).

## Data Flow Pattern

The LOGADDEXP data flow has **5 stages** due to LHS/RHS preprocessing and post activation:

1. **Reader kernel** reads tile from input A into `cb_pre_lhs` (CB c_0) and tile from input B into `cb_pre_rhs` (CB c_1), one tile at a time via NoC async read.

2. **Compute kernel - LHS Preprocessing (EXP)**: The `PREPROCESS(LHS, ...)` macro fires because `HAS_ACTIVATIONS(LHS)` is true. It:
   - Waits for a tile in `cb_pre_lhs` (c_0)
   - Reserves space in `cb_post_lhs` (c_3)
   - Copies tile to DEST, applies `exp_tile_init()` + `exp_tile(i)` via `PROCESS_LHS_ACTIVATIONS(i)`
   - Packs result into `cb_post_lhs` (c_3)
   - Pops `cb_pre_lhs`, pushes `cb_post_lhs`

3. **Compute kernel - RHS Preprocessing (EXP)**: Same pattern as LHS, but for input B:
   - Reads from `cb_pre_rhs` (c_1) -> applies exp -> writes to `cb_post_rhs` (c_4)

4. **Compute kernel - Binary ADD + LOG Postprocessing**: The main binary compute loop:
   - Waits for tiles in `cb_post_lhs` (c_3) and `cb_post_rhs` (c_4)
   - Copies both tiles into DEST registers (LHS at slot `i*2`, RHS at slot `i*2+1`)
   - Because `HAS_ACTIVATIONS(POST)` is true, `BINARY_SFPU_INIT` runs inside the inner loop (re-initializes for ADD each iteration)
   - Calls `BINARY_SFPU_OP(i*2, i*2+1, i*2)` which resolves to `add_binary_tile(i*2, i*2+1, i*2)`
   - Calls `PROCESS_POST_ACTIVATIONS(i*2)` which resolves to `log_tile_init(); log_tile(i*2);`
   - Packs result from DEST to `cb_out` (c_2)

5. **Writer kernel** reads tiles from `cb_out` (c_2) and writes them to the output buffer in DRAM/L1 via NoC async write.

## Circular Buffer Configuration

| CB ID | Name/Purpose | Data Format | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|
| c_0 | `cb_pre_lhs` - Raw input A | a_data_format | 2 (interleaved) or shard_volume | 1 | Double (interleaved) | Reader | Compute (EXP preprocess) |
| c_1 | `cb_pre_rhs` - Raw input B | b_data_format | 2 (interleaved) or shard_volume | 1 | Double (interleaved) | Reader | Compute (EXP preprocess) |
| c_2 | `cb_out` - Final output | c_data_format | 2 (interleaved) or shard_volume | 1 | Double (interleaved) | Compute (LOG post) | Writer |
| c_3 | `cb_post_lhs` - EXP(A) intermediate | a_data_format (SFPU path) | 1 | 1 | Single | Compute (EXP LHS) | Compute (ADD binary) |
| c_4 | `cb_post_rhs` - EXP(B) intermediate | b_data_format (SFPU path) | 1 | 1 | Single | Compute (EXP RHS) | Compute (ADD binary) |

**Note on intermediate format**: For LOGADDEXP, since `op_has_exp` is true (line 631-632 of program factory) AND `is_sfpu_op` is true, the intermediate CBs (c_3 and c_4) use `a_data_format` / `b_data_format` respectively (not Float16_b). The Float16_b intermediate format is only used for the FPU path.

## Pipeline Pattern Summary

- **c_0 and c_1**: Double-buffered (capacity=2, block=1) in interleaved mode, allowing reader to fill the next tile while compute processes the current one.
- **c_2**: Double-buffered (capacity=2, block=1) in interleaved mode, allowing compute to produce the next result while writer drains the current one.
- **c_3 and c_4**: Single-buffered (capacity=1, block=1). These are intra-compute intermediates that do not overlap with reader/writer. The compute kernel serializes: preprocess LHS -> preprocess RHS -> binary op + post -> pack.

## Index Calculations

The reader and writer kernels use a 5D nested loop structure to map logical tensor coordinates to physical tile offsets:

- **Dimensions**: `nD` (collapsed dims > 5), `D` (dim -5), `N` (dim -4), `C` (dim -3), `Ht` (height in tiles), `Wt` (width in tiles)
- **Start tile decomposition**: The `start_tile_id` is decomposed into per-dimension offsets using modular arithmetic
- **Stride computation for broadcast**: Per-input strides are computed as `aHt * aWt * aC * aN * aD * (aND > 1)` etc., where the `(dim > 1)` factor zeroes the stride when a dimension is broadcast (size 1)
- **TensorAccessor**: Used for DRAM-interleaved access, abstracting bank mapping. Compile-time and common runtime args are appended via `TensorAccessorArgs`.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile reads with broadcast-aware striding. Both A and B tiles are read one at a time with `noc_async_read_page` followed by a barrier. The reader reads A and B tiles in lockstep for the no-broadcast case.
- **Sharded**: If both inputs are sharded, the reader simply does `cb_reserve_back` + `cb_push_back` on the full shard volume (the CB is backed by the sharded buffer directly).

### Write Pattern

- **Interleaved**: Sequential tile writes with `noc_async_write_page` followed by a barrier, one tile at a time.
- **Sharded**: No explicit writes needed; the output CB is backed by the sharded output buffer.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular (when zero_start_grid) or arbitrary CoreRangeSet |
| Work Splitting | `split_work_to_cores` divides total output tiles across cores |
| Core Group 1 | Gets `num_tiles_per_core_group_1` tiles (ceil division) |
| Core Group 2 | Gets `num_tiles_per_core_group_2` tiles (remainder cores, 1 fewer tile) |
| No-op Cores | Cores outside both groups receive all-zero runtime args and exit immediately |
| Row-major traversal | Default; cores are enumerated row-major within the grid |

For sharded tensors, the core grid is determined by the shard spec, and each core processes exactly its shard volume.

## Arguments

### Compile-Time Arguments

**Compute Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `num_tiles_per_cycle` | uint32_t | Always 1 - tiles produced per read-compute-write cycle |

**Compute Kernel Defines** (compile-time via preprocessor):

| Define | Value for LOGADDEXP | Description |
|---|---|---|
| `BINARY_SFPU_INIT` | `add_binary_tile_init();` | Initializes SFPU for ADD operation |
| `BINARY_SFPU_OP` | `add_binary_tile` | The core SFPU binary function |
| `PROCESS_LHS_ACTIVATIONS(i)` | `exp_tile_init(); exp_tile(i);` | EXP applied to LHS |
| `PROCESS_RHS_ACTIVATIONS(i)` | `exp_tile_init(); exp_tile(i);` | EXP applied to RHS |
| `PROCESS_POST_ACTIVATIONS(i)` | `log_tile_init(); log_tile(i);` | LOG applied to result |
| `BCAST_INPUT` | `""` (no broadcast) or `"0"`/`"1"` | Which input is broadcast |

**Reader Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessorArgs(A) | uint32_t[] | Compile-time tensor accessor args for input A |
| N+1..M | TensorAccessorArgs(B) | uint32_t[] | Compile-time tensor accessor args for input B |
| M+1 | has_sharding | uint32_t | 1 if native L1 sharding active, 0 otherwise |

**Writer Kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0..N | TensorAccessorArgs(C) | uint32_t[] | Compile-time tensor accessor args for output C |
| N+1 | has_sharding | uint32_t | 1 if native L1 sharding active, 0 otherwise |

### Runtime Arguments

**Reader Kernel** (21 args per core):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Base address of input A buffer |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core |
| 2 | a_num_tiles | uint32_t | Number of A shard tiles (sharded only) |
| 3 | c_num_tiles | uint32_t | Number of output tiles for this core |
| 4 | c_current_shard_width | uint32_t | Width of current shard in tiles (sharded only) |
| 5 | nD_stride | uint32_t | A's stride for collapsed nD dimension |
| 6 | d_stride | uint32_t | A's stride for D dimension |
| 7 | n_stride | uint32_t | A's stride for N dimension |
| 8 | c_stride | uint32_t | A's stride for C dimension |
| 9 | D | uint32_t | Output D dimension |
| 10 | N | uint32_t | Output N dimension |
| 11 | C | uint32_t | Output C dimension |
| 12 | Ht | uint32_t | Output height in tiles |
| 13 | Wt | uint32_t | Output width in tiles |
| 14 | cND | uint32_t | Output collapsed nD dimension |
| 15 | src_addr_b | uint32_t | Base address of input B buffer |
| 16 | nD_stride_b | uint32_t | B's stride for collapsed nD dimension |
| 17 | d_stride_b | uint32_t | B's stride for D dimension |
| 18 | n_stride_b | uint32_t | B's stride for N dimension |
| 19 | c_stride_b | uint32_t | B's stride for C dimension |
| 20 | b_num_tiles | uint32_t | Number of B shard tiles (sharded only) |

**Writer Kernel** (11 args per core):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output C buffer |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Number of output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles (sharded only) |
| 4 | D | uint32_t | Output D dimension |
| 5 | N | uint32_t | Output N dimension |
| 6 | C | uint32_t | Output C dimension |
| 7 | Ht | uint32_t | Output height in tiles |
| 8 | Wt | uint32_t | Output width in tiles |
| 9 | cND | uint32_t | Output collapsed nD dimension |
| 10 | (unused) | uint32_t | Reserved (set to 0) |

**Compute Kernel** (4 args per core):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Number of tiles to process on this core |
| 1 | freq | uint32_t | Broadcast repetition frequency (1 for NONE) |
| 2 | counter | uint32_t | Broadcast starting counter (0 for NONE) |
| 3 | compute_scalar_value | uint32_t | Unused for LOGADDEXP (0) |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` (for SubtileBroadcastType::NONE; other broadcast types use corresponding reader variants)
- **Key Logic**: Reads both A and B tiles in a single nested 5D loop. For each output tile position, reads one tile from A into c_0 and one tile from B into c_1, respecting broadcast strides. Uses TensorAccessor for DRAM bank-interleaved addressing. Sharded inputs skip the read loop and just reserve/push the full shard.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Iterates through output tiles in the same 5D nested loop order, waiting for each tile in c_2 from the compute kernel, then writing it to the output buffer via NoC. Sharded outputs skip the write loop entirely.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`

(Selected when `SubtileBroadcastType::NONE` and `is_sfpu=true`; other broadcast types use `eltwise_binary_sfpu.cpp` or the `kernels_ng` row/col bcast variants.)

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
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    // Runtime arg 0: total number of tiles this core must process

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    // Compile-time arg 0: always 1 for this operation (1 tile per cycle)

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    // CB c_0: raw input A tiles from reader
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    // CB c_1: raw input B tiles from reader
    constexpr auto cb_out = tt::CBIndex::c_2;
    // CB c_2: final output tiles to writer

    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    // For LOGADDEXP: HAS_ACTIVATIONS(LHS) is true (EXP), so cb_post_lhs = c_3
    // c_3 holds exp(A) intermediate results
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;
    // For LOGADDEXP: HAS_ACTIVATIONS(RHS) is true (EXP), so cb_post_rhs = c_4
    // c_4 holds exp(B) intermediate results

    unary_op_init_common(cb_post_lhs, cb_out);
    // Initializes unpack/pack config for the compute pipeline
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
    // PACK_RELU is not defined for LOGADDEXP

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
    // This branch is NOT taken for LOGADDEXP since all three activation stages are active.
    // BINARY_SFPU_INIT would expand to add_binary_tile_init() but is deferred to the inner loop.
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Main tile processing loop - one tile per iteration

        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        // Expands to PREPROCESS_1 macro because HAS_ACTIVATIONS(LHS)=1:
        // 1. Reconfigures pack format from cb_out to cb_post_lhs (c_3)
        // 2. Waits for 1 tile in cb_pre_lhs (c_0) from reader
        // 3. Reserves 1 slot in cb_post_lhs (c_3)
        // 4. Acquires DEST registers
        // 5. Copies tile from c_0 to DEST[0]
        // 6. Calls PROCESS_LHS_ACTIVATIONS(0) which expands to: exp_tile_init(); exp_tile(0);
        //    This computes exp(A) in DEST[0] using SFPU
        // 7. Commits DEST, waits, packs DEST[0] into c_3
        // 8. Releases DEST
        // 9. Pops c_0, pushes c_3
        // 10. Reconfigures pack format back from c_3 to cb_out

        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);
        // Wait for the preprocessed exp(A) tile to be available in c_3

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        // Same pattern as LHS but for input B:
        // Reads from c_1, applies exp_tile via PROCESS_RHS_ACTIVATIONS(0),
        // writes exp(B) into c_4

        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);
        // Wait for the preprocessed exp(B) tile in c_4

        cb_reserve_back(cb_out, num_tiles_per_cycle);
        // Reserve space in the output CB c_2 for the final result

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
        // NOT taken for LOGADDEXP because HAS_ACTIVATIONS(POST) is also true
#endif
        tile_regs_acquire();
        // Acquire exclusive access to DEST registers for the binary op phase

        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        // Initialize unpacker for cb_post_lhs (c_3) with data type reconfiguration
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);
            // Copy exp(A) tile from c_3 position i into DEST register slot 0 (i*2 = 0)
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        // Reconfigure unpacker for cb_post_rhs (c_4)
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);
            // Copy exp(B) tile from c_4 position i into DEST register slot 1 (i*2+1 = 1)

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
            // For LOGADDEXP: expands to add_binary_tile_init();
            // Re-initializes SFPU for ADD before every tile because LOG postprocessing
            // may have changed SFPU configuration
#endif
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
            // Expands to: add_binary_tile(0, 1, 0)
            // Performs SFPU element-wise addition: DEST[0] = DEST[0] + DEST[1]
            // Result (exp(A) + exp(B)) is written to DEST[0]

            PROCESS_POST_ACTIVATIONS(i * 2);
            // Expands to: log_tile_init(); log_tile(0);
            // Computes natural log of DEST[0] in-place: DEST[0] = log(exp(A) + exp(B))
        }
        tile_regs_commit();
        // Signal that DEST register writes are complete

        tile_regs_wait();
        // Wait for DEST data to be ready for packing

        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);
            // Pack final result from DEST[0] into output CB c_2
        }
        tile_regs_release();
        // Release DEST registers

        cb_push_back(cb_out, num_tiles_per_cycle);
        // Signal writer that 1 output tile is ready in c_2
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle);
        // Free the exp(A) intermediate tile in c_3
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
        // Free the exp(B) intermediate tile in c_4
    }
}
```

### SFPU Kernel Implementation

LOGADDEXP uses three SFPU operations in sequence. The core binary operation is SFPU ADD; EXP and LOG are unary SFPU operations applied as preprocessing and postprocessing.

#### SFPU Kernel File (ADD)

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h`
(identical implementation for `wormhole_b0`)

#### Annotated SFPU Kernel Source (ADD)

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

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This implements the "add 0x7fff + LSB" algorithm for correct tie-breaking
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Get the float32 bits as unsigned integer
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32)
    // This is needed for the tie-breaker: round to even
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - If lower 16 bits > 0x8000: overflow -> rounds up
    // - If lower 16 bits < 0x8000: no overflow -> rounds down
    // - If lower 16 bits = 0x8000 (tie) and lsb=0: no overflow -> stays even
    // - If lower 16 bits = 0x8000 (tie) and lsb=1: overflow -> rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits
    bits = bits & 0xFFFF0000U;

    // Reinterpret back as float
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

// Main SFPU binary operation kernel - processes one tile face per iteration
// For LOGADDEXP, this is instantiated with BINOP=BinaryOp::ADD
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
    // Delegates to the tt_llk implementation (see DeepWiki-sourced code below)
}

// The _calculate_sfpu_binary_ implementation (from tt_llk submodule):
// template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
// inline void _calculate_sfpu_binary_(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
//     for (int d = 0; d < ITERATIONS; d++) {
//         constexpr uint32_t dst_tile_size_sfpi = 32;
//         // Each tile in DEST occupies 32 rows when accessed via SFPI (64/SFP_DESTREG_STRIDE)
//
//         sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
//         // Load 32 elements (one SFPU lane width) from DEST[in0_tile * 32 + d]
//
//         sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
//         // Load 32 elements from DEST[in1_tile * 32 + d]
//
//         sfpi::vFloat result = 0.0f;
//
//         if constexpr (BINOP == BinaryOp::ADD) {
//             result = in0 + in1;
//             // SFPU vector addition: result[lane] = in0[lane] + in1[lane]
//         }
//         // ... other BINOP cases (SUB, MUL, DIV, RSUB, etc.)
//
//         sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
//         // Write result back to DEST[out_tile * 32 + d]
//
//         sfpi::dst_reg++;
//         // Advance to next row (next set of 32 elements)
//     }
// }

// Initialization for SFPU binary ops
template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
    // For ADD: no special initialization needed (init is a no-op for ADD/SUB)
    // For DIV/POW: would call _init_sfpu_reciprocal_<false>()
    // For XLOGY: would call _init_log_<APPROXIMATION_MODE>()
}

// ... mul and div variants omitted (not used by LOGADDEXP)

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description | Used In |
|---|---|---|
| `sfpi::dst_reg[index]` (load) | Loads a vector of 32 float elements from DEST register file at the given offset | ADD kernel - loads operands |
| `sfpi::dst_reg[index]` (store) | Stores a vector of 32 float elements back to DEST register file | ADD kernel - stores result |
| `sfpi::dst_reg++` | Advances the DEST register pointer by one row (32 elements) for next iteration | ADD kernel - iterates over tile faces |
| `in0 + in1` (vFloat addition) | SFPU vector floating-point addition on 32 lanes simultaneously | ADD kernel - core operation |
| `exp_tile(idst)` | Computes element-wise exponential using SFPU `calculate_exponential` function | EXP preprocessing |
| `log_tile(idst)` | Computes element-wise natural logarithm using SFPU `llk_math_eltwise_unary_sfpu_log` | LOG postprocessing |
| `copy_tile(cb, idx, dst)` | Unpacks a tile from circular buffer into DEST register at specified slot | Loading tiles for binary op |

#### SFPU Register Usage

- **DEST[0]** (slot `i*2` = 0): Holds input A during EXP preprocessing, then holds exp(A) during the binary phase, then holds exp(A)+exp(B) after ADD, and finally holds log(exp(A)+exp(B)) after LOG postprocessing.
- **DEST[1]** (slot `i*2+1` = 1): Holds exp(B) during the binary ADD phase. Consumed by ADD but not written to.
- **SFPU internal registers**: The `vFloat` variables (`in0`, `in1`, `result`) map to SFPU vector registers that process 32 elements per cycle. The SFPU processes 8 iterations (ITERATIONS=8) of 32 elements each, covering all 256 elements of a tile face pair (4 faces x 32 rows / face x 2 = 256 row iterations over a 32x32 tile mapped to 8 iterations of 32-element vectors).

#### SFPU Execution Flow

1. **EXP Preprocessing (per input)**:
   - `copy_tile_to_dst_init_short(cb_pre)` - configure unpacker for the input CB
   - `copy_tile(cb_pre, 0, 0)` - unpack tile from CB into DEST[0]
   - `exp_tile_init()` - configure SFPU for exponential computation
   - `exp_tile(0)` - compute exp() on DEST[0] in-place, iterating 8 times over 32-element vectors
   - `pack_tile(0, cb_post)` - pack result from DEST[0] into intermediate CB

2. **Binary ADD + LOG Postprocessing**:
   - `copy_tile(cb_post_lhs, 0, 0)` - unpack exp(A) from c_3 into DEST[0]
   - `copy_tile(cb_post_rhs, 0, 1)` - unpack exp(B) from c_4 into DEST[1]
   - `add_binary_tile_init()` - configure SFPU for addition (re-init because LOG may have changed state)
   - `add_binary_tile(0, 1, 0)` - compute DEST[0] = DEST[0] + DEST[1] via `_llk_math_eltwise_binary_sfpu_params_` which calls `calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8>`. Internally iterates 8 times, each processing 32 elements from both DEST[0] and DEST[1].
   - `log_tile_init()` - configure SFPU for natural logarithm
   - `log_tile(0)` - compute ln() on DEST[0] in-place
   - `pack_tile(0, cb_out)` - pack final result into output CB

#### SFPU Configuration

- **UnpackToDestMode**: Set to `UnpackToDestFp32` for all source CBs (c_0, c_1, c_3, c_4) when `is_sfpu_op` is true and `op_type != POWER`. This ensures data is unpacked to full FP32 precision in DEST registers for SFPU processing.
- **fp32_dest_acc_en**: Enabled when output format is Float32, or when both inputs are Float32. Controls DEST register accumulator precision.
- **APPROX**: Template parameter on SFPU functions, controlled by `ComputeConfig`. Affects exp and log approximation quality.
- **fast_and_approx**: The `exp_tile` and `log_tile` calls use default template parameters (`approx=false`, `fast_and_approx=true` for exp; `fast_and_approx=false` for log when called without explicit template args from the activation define system).

#### Hardware Compatibility Notes

- The `_calculate_sfpu_binary_` and `_sfpu_binary_init_` implementations are identical between Wormhole B0 and Blackhole architectures for the ADD operation.
- Both architectures support the same SFPI vector operations (`vFloat` addition, DEST register access patterns).
- The EXP and LOG SFPU kernels also share implementations across architectures, though specific approximation accuracy may differ due to LUT contents.

## Implementation Notes

1. **Composite Operation Overhead**: LOGADDEXP is one of the most expensive binary_ng SFPU operations because it requires three SFPU operations per tile (EXP on A, EXP on B, ADD, LOG on result). Each tile passes through the DEST registers four times (EXP_A, EXP_B, ADD+LOG).

2. **SFPU Re-initialization**: Because LOGADDEXP has all three activation stages active (LHS, RHS, POST), the `BINARY_SFPU_INIT` (add_binary_tile_init) is placed inside the innermost loop, being called once per tile. Similarly, `log_tile_init()` and `exp_tile_init()` are called per tile. This repeated initialization adds overhead but is necessary to reconfigure the SFPU pipeline between different operations.

3. **Intermediate CB Sizing**: CBs c_3 and c_4 are single-buffered (capacity=1) because they serve as intra-compute intermediates. There is no producer-consumer overlap possible - the same compute kernel produces and consumes these buffers.

4. **Pack Format Reconfiguration**: The PREPROCESS macro includes `pack_reconfig_data_format` calls to switch the packer between the intermediate format (c_3/c_4) and the output format (c_2). This is necessary because the intermediate and output may have different data formats.

5. **Broadcast Handling**: For broadcast cases (e.g., scalar + tensor), different reader and compute kernel variants are selected. The LOGADDEXP operation supports all SubtileBroadcastType variants through the binary_ng framework's kernel selection mechanism.

6. **op_has_exp Flag**: The program factory explicitly checks for LOGADDEXP (line 631-632) to set `op_has_exp = true`. This flag affects the intermediate data format selection: for the FPU path it forces Float16_b intermediates; for the SFPU path it uses the native input format.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What are the different subtypes (scalar, bcast, none) and how does it handle SFPU vs FPU compute kernels?"
   **Reason**: Needed to understand the overall architecture of the binary_ng framework before diving into code.
   **Key Findings**: The framework selects reader/compute/writer kernels based on SubtileBroadcastType and is_sfpu flag. SFPU operations use `eltwise_binary_sfpu_*.cpp` compute kernels while FPU operations use `eltwise_binary_*.cpp`. The OpConfig class encapsulates the binary op type and generates preprocessor defines.

2. **Query**: "What is the logaddexp operation in TTNN? How is it implemented as a binary_ng operation?"
   **Reason**: Needed to confirm the decomposition of LOGADDEXP into component operations.
   **Key Findings**: Confirmed that LOGADDEXP decomposes into EXP(LHS), EXP(RHS), ADD, LOG. It is an SFPU operation when inputs are FLOAT32.

3. **Query**: "Show the full source code of _calculate_sfpu_binary_ and _sfpu_binary_init_ functions from tt_llk."
   **Reason**: The tt_llk submodule was empty in this worktree, so DeepWiki was the only way to access the underlying SFPU binary kernel implementation.
   **Key Findings**: The ADD case is a simple `in0 + in1` vector operation. The function iterates 8 times over 32-element vectors, covering the full tile. The init function is a no-op for ADD (only DIV/POW/XLOGY need initialization).

4. **Query**: "What is _llk_math_eltwise_binary_sfpu_params_ and how does it coordinate SFPU binary operations?"
   **Reason**: Needed to understand the dispatch layer between the high-level API and the SFPU kernel.
   **Key Findings**: It is a generic dispatcher that handles DEST register setup, calls the SFPU function, and manages cleanup. It resides in the tt_llk submodule.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 226-231)
   **Reason**: Authoritative source for LOGADDEXP's OpConfig decomposition
   **Key Information**: `process_lhs=EXP, process_rhs=EXP, binary_op=ADD, postprocess=LOG`

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` (lines 247-250, 587, 634)
   **Reason**: Needed to determine the exact SFPU function names generated for EXP and LOG activations
   **Key Information**: EXP generates `exp_tile_init(); exp_tile(i);` and LOG generates `log_tile_init(); log_tile(i);`

3. **Source**: `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h`
   **Reason**: Needed to understand the add_binary_tile API and its delegation to LLK
   **Key Information**: `add_binary_tile` calls `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>` which dispatches to `calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8>`

4. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Needed to understand exp_tile's template parameters and SFPU dispatch
   **Key Information**: exp_tile dispatches to `calculate_exponential` with configurable approximation modes, scaling, and input clamping
