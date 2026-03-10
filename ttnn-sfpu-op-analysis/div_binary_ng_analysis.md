# DIV (binary_ng) Implementation Analysis

## Overview
The DIV operation computes element-wise floating-point division: `c = a / b`. It is part of the `binary_ng` ("next generation") binary operation framework in TTNN. When used as an SFPU operation (which is the case for FLOAT32/FLOAT32 inputs or INT32/INT32 inputs), it dispatches to a dedicated SFPU division kernel (`div_binary_tile`) that computes division via reciprocal approximation followed by multiplication. For non-SFPU paths (e.g., BFLOAT16), DIV is decomposed into `RECIP(b) * a` using the FPU MUL pipeline with a unary reciprocal pre-processing step on the RHS.

This analysis focuses on the **SFPU path** (floating-point DIV) and the **INT32 path** (`div_int32_tile`).

**Program Factory Path**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

## Work Unit Definition
One work unit is **one tile** (32x32 elements). The compute kernel processes `num_tiles_per_cycle = 1` tile per read-compute-write cycle. Each cycle loads one LHS tile and one RHS tile into DEST registers, performs the SFPU division, and packs the result to the output circular buffer.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Tensor A (LHS) | Tensor B (RHS) |
|---|---|---|
| Dimension Convention | ND, D, N, C, H, W (dims > 5 collapsed into ND) | Same as A, with broadcasting |
| Tensor Layout | TILED (32x32) | TILED (32x32) |
| Memory Layout | Interleaved (DRAM/L1) or Sharded (L1) | Interleaved (DRAM/L1) or Sharded (L1), or scalar |
| Buffer Type | DRAM or L1 | DRAM or L1 |
| Data Type | FLOAT32, INT32, BFLOAT16 | Same as A (or scalar) |

When B is a scalar (not a tensor), it is packed into a single tile by the writer kernel using `fill_with_val` and placed into CB c_1.

### Output Tensor(s)

| Property | Tensor C (Output) |
|---|---|
| Dimension Convention | ND, D, N, C, H, W (broadcast-expanded) |
| Tensor Layout | TILED (32x32) |
| Memory Layout | Interleaved (DRAM/L1) or Sharded (L1) |
| Buffer Type | DRAM or L1 |
| Data Type | Same as input dtype (FLOAT32, INT32, BFLOAT16) |

### Layout Transformations
No tilize/untilize conversions occur within the program factory. Input tensors must already be in tiled layout. When SFPU path is active and the operation is not POWER, `UnpackToDestMode::UnpackToDestFp32` is enabled for all source CBs, meaning data is unpacked to FP32 in DEST registers regardless of the input format.

## Data Flow Pattern

### Two-Tensor Path (A op B, both tensors)
1. **Reader kernel** (ReaderNoBcastNg) reads tiles from both tensor A and tensor B into CB c_0 and CB c_1, handling broadcasting via stride manipulation
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1, copies them to DEST registers, executes `BINARY_SFPU_OP` (which is `div_binary_tile`), and packs the result to CB c_2
3. **Writer kernel** (WriterNoBcastNg) reads tiles from CB c_2 and writes them to the output tensor

### Scalar Path (A op scalar)
1. **Reader kernel** (ReaderNoBcast) reads tiles from tensor A into CB c_0
2. **Writer kernel** (WriterScalar) fills a single tile in CB c_1 with the scalar value, then writes output tiles from CB c_2 to DRAM
3. **Compute kernel** (ComputeScalar) waits for the scalar tile in CB c_1 once, then for each LHS tile: copies both to DEST, executes `BINARY_SFPU_OP`, packs to CB c_2

### Broadcast Path (various SubtileBroadcastTypes)
The broadcast input is loaded once and reused for multiple tiles of the other input. The `process_tile` function in `eltwise_binary_sfpu.cpp` handles this with a nested loop structure controlled by `tile_freq` and `tile_start` runtime arguments.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity (tiles) | Data Format | Producer | Consumer | Buffering |
|---|---|---|---|---|---|---|---|
| c_0 | cb_pre_lhs / cb_src_a | Input A tiles | 2 (interleaved) or shard volume (sharded) | A's data format | Reader | Compute | Double-buffered (interleaved) |
| c_1 | cb_pre_rhs / cb_src_b | Input B tiles or scalar tile | 2 (tensor) or 1 (scalar) or shard volume (sharded) | B's data format | Reader/Writer(scalar) | Compute | Double-buffered (tensor) / Single-buffered (scalar) |
| c_2 | cb_out / cb_dst | Output C tiles | 2 (interleaved) or shard volume (sharded) | C's data format | Compute | Writer | Double-buffered (interleaved) |
| c_3 | cb_post_lhs | LHS after activation preprocessing | 1 | A's data format | Compute (preprocess) | Compute (main) | Single-buffered (only if LHS activations present) |
| c_4 | cb_post_rhs | RHS after activation preprocessing | 1 | B's data format | Compute (preprocess) | Compute (main) | Single-buffered (only if RHS activations present) |
| c_5 | Row bcast A scratch | Scratch for ROW_A or ROW_A_COL_B bcast | 2 | A's data format | Reader | Compute | Double-buffered (only if applicable bcast type) |
| c_6 | Row bcast B scratch | Scratch for ROW_B or ROW_B_COL_A bcast | 2 | B's data format | Reader | Compute | Double-buffered (only if applicable bcast type) |

For the standard DIV operation with no pre/post activations, only CB c_0, c_1, and c_2 are active. CB c_3 and c_4 are unused since DIV has no `process_lhs` or `process_rhs` defined in `OpConfig`.

## Pipeline Pattern Summary
- **Interleaved mode**: CB c_0 and c_1 have capacity=2 (double-buffered), allowing the reader to fill the next tile while compute processes the current one. CB c_2 has capacity=2 (double-buffered), allowing compute to produce while writer drains.
- **Sharded mode**: CB capacities match shard volumes. All tiles are pre-loaded by the reader via `cb_reserve_back`/`cb_push_back` on the sharded buffer, then consumed sequentially by compute.
- **Scalar mode**: CB c_1 holds exactly 1 tile (the scalar), loaded once and never popped until all output tiles are produced.

## Index Calculations
The reader kernel computes a multi-dimensional tile offset from the output `start_tile_id`:
- `start_tile_id` is decomposed into `(nd, d, n, c, th, tw)` coordinates based on the output tensor shape `(cND, D, N, C, Ht, Wt)`
- Each input tensor has its own stride per dimension, computed as `dim_size * (dim_size > 1)` -- this cleverly handles broadcasting by setting the stride to 0 when a dimension has size 1
- The input tile offset is: `nd * nD_stride + d * d_stride + n * n_stride + c * c_stride + th * Wt + tw`

For sharded tensors, `dst_shard_width` limits the `tw` loop to only iterate over the tiles within the shard.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Tiles are read one at a time via `noc_async_read_page` with barrier after each tile. The iteration order is ND-major: nd -> d -> n -> c -> th -> tw (innermost). For broadcasting, the input stride is 0 in broadcast dimensions, causing the same tile to be re-read.
- **Sharded**: All tiles are already in L1. The reader simply marks them available via `cb_reserve_back`/`cb_push_back`.

### Write Pattern
- **Interleaved**: Tiles written one at a time via `noc_async_write_page` with barrier, following the same ND-major iteration order.
- **Sharded**: Output buffer is already in L1 at the correct address. Writer is a no-op (tiles stay in the sharded CB).

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular grid from `operation_attributes.worker_grid` |
| Work Splitting | Output tiles distributed evenly across cores via `split_work_to_cores` |
| Load Balancing | Two core groups: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets the remainder |
| Remainder Handling | Cores not in group 1 or group 2 receive zero-tile runtime args and are effectively no-ops |
| Sharded Mode | Core grid determined by shard spec; each core processes its own shard |
| Zero-Start Optimization | When the grid starts at (0,0) and is a single rectangle, a faster work distribution algorithm is used |

## Arguments

### Compile-Time Arguments

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1 -- number of output tiles produced per compute cycle |
| (via defines) | BINARY_SFPU_INIT | string | `div_binary_tile_init();` -- SFPU init macro for reciprocal LUT |
| (via defines) | BINARY_SFPU_OP | string | `div_binary_tile` -- SFPU operation function name |
| (via defines) | BCAST_INPUT | string | `""` (no bcast) or `"0"` (bcast A) or `"1"` (bcast B) |
| (reader CT) | TensorAccessor args | varies | Compile-time portion of TensorAccessor for A and B |
| (reader CT) | has_sharding | uint32_t | 0 or 1 -- whether sharded mode is active |
| (writer CT) | TensorAccessor args | varies | Compile-time portion of TensorAccessor for C |
| (writer CT) | has_sharding | uint32_t | 0 or 1 |

### Runtime Arguments

**Reader** (21 args for two-tensor path):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | DRAM address of tensor A |
| 1 | start_tile_id | uint32_t | Starting output tile ID for this core |
| 2 | src_num_tiles (a) | uint32_t | Number of A tiles in shard (sharded only) |
| 3 | dst_num_tiles | uint32_t | Total output tiles assigned to this core |
| 4 | dst_shard_width | uint32_t | Shard width in tiles (sharded only, 0 otherwise) |
| 5-8 | nD/d/n/c strides (A) | uint32_t | Broadcast-aware strides for tensor A |
| 9-14 | D, N, C, Ht, Wt, cND | uint32_t | Output tensor shape dimensions |
| 15 | src_addr_b | uint32_t | DRAM address of tensor B |
| 16-19 | nD/d/n/c strides (B) | uint32_t | Broadcast-aware strides for tensor B |
| 20 | src_num_tiles_b | uint32_t | Number of B tiles in shard (sharded only) |

**Writer** (11 args for two-tensor path):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM address of output tensor C |
| 1 | start_tile_id | uint32_t | Starting output tile ID |
| 2 | dst_num_tiles | uint32_t | Total output tiles for this core |
| 3 | dst_shard_width | uint32_t | Shard width in tiles |
| 4-10 | D, N, C, Ht, Wt, cND, 0 | uint32_t | Output shape dimensions + padding |

**Compute** (4 args):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Total output tiles for this core |
| 1 | tile_freq | uint32_t | Broadcast repeat frequency (1 for no bcast, Ht*Wt for scalar bcast, Wt for col bcast) |
| 2 | tile_start | uint32_t | Starting offset within the broadcast cycle |
| 3 | compute_scalar_value | uint32_t | Unused for standard DIV (set to 0) |

## Kernel Implementations

### Reader Kernel (Two-Tensor Path)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads tiles for both A and B tensors in lockstep. Uses TensorAccessor for DRAM page addressing. Supports sharded inputs via conditional compilation (`SRC_SHARDED`, `SRC_SHARDED_B`). Broadcasting is handled entirely through stride values -- when a dimension is broadcast (size 1), its stride is set to 0, causing the same tiles to be re-read.

### Reader Kernel (Scalar Path)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/reader_interleaved_no_bcast.cpp`
- **Key Logic**: Reads only tensor A tiles. The scalar is handled by the writer kernel.

### Writer Kernel (Two-Tensor Path)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/dataflow/writer_interleaved_no_bcast.cpp`
- **Key Logic**: Writes output tiles from CB c_2 to DRAM. For sharded outputs, the writer is effectively a no-op since data is already in L1.

### Writer Kernel (Scalar Path)
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/writer_interleaved_scalar.cpp`
- **Key Logic**: Fills a single tile in CB c_1 with the packed scalar value, then writes output tiles from CB c_2 to DRAM. The scalar fill uses `fill_with_val<1024, float>` for FLOAT32 or `fill_with_val_bfloat16` for BFLOAT16.

### Compute Kernel

#### Compute Kernel File (No Broadcast -- primary path)
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
#include "api/compute/binary_max_min.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/binary_comp.h"

#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);  // runtime arg: total tiles this core must process

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // always 1

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;   // input A circular buffer
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;   // input B circular buffer
    constexpr auto cb_out = tt::CBIndex::c_2;        // output C circular buffer

    // If LHS/RHS activations are defined, use intermediate CBs c_3/c_4; otherwise alias to pre CBs
    constexpr auto cb_post_lhs = HAS_ACTIVATIONS(LHS) ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = HAS_ACTIVATIONS(RHS) ? tt::CBIndex::c_4 : cb_pre_rhs;

    unary_op_init_common(cb_post_lhs, cb_out);  // initialize unpack/pack pipeline for LHS->OUT path
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // configure packer for ReLU clamping if enabled
#endif

    // For DIV with no activations: BINARY_SFPU_INIT expands to div_binary_tile_init()
    // which calls _sfpu_binary_init_<APPROX, BinaryOp::DIV>() to initialize reciprocal LUT
#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT  // div_binary_tile_init(); -- sets up SFPU reciprocal lookup tables
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // PREPROCESS macros are no-ops for DIV (no lhs/rhs activations)
        PREPROCESS(LHS, cb_pre_lhs, cb_post_lhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_lhs, num_tiles_per_cycle);  // wait for 1 LHS tile from reader

        PREPROCESS(RHS, cb_pre_rhs, cb_post_rhs, cb_out, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle);  // wait for 1 RHS tile from reader

        cb_reserve_back(cb_out, num_tiles_per_cycle);  // reserve space in output CB for 1 tile

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
        BINARY_SFPU_INIT
#endif
        tile_regs_acquire();  // acquire exclusive access to DEST register file

        // Copy LHS tile to DEST[0] (even index)
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);  // configure unpacker for LHS format
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);  // copy tile 0 from CB to DEST[0]
        }

        // Copy RHS tile to DEST[1] (odd index)
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);  // reconfigure unpacker for RHS format
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);  // copy tile 0 from CB to DEST[1]

#if HAS_ACTIVATIONS(POST)
            BINARY_SFPU_INIT
#endif
            // BINARY_SFPU_OP expands to: div_binary_tile(0, 1, 0)
            // Performs DEST[0] = DEST[0] / DEST[1] using SFPU reciprocal + multiply
            BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);

            // PROCESS_POST_ACTIVATIONS is empty for plain DIV (no post-processing)
            PROCESS_POST_ACTIVATIONS(i * 2);
        }
        tile_regs_commit();  // signal that DEST registers are ready for packing

        tile_regs_wait();  // wait for pack stage to be ready
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 2, cb_out);  // pack DEST[0] result into output CB c_2
        }
        tile_regs_release();  // release DEST registers

        cb_push_back(cb_out, num_tiles_per_cycle);     // publish output tile to writer
        cb_pop_front(cb_post_lhs, num_tiles_per_cycle); // free consumed LHS tile
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle); // free consumed RHS tile
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (contains `calculate_sfpu_binary_div`)
- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (identical implementation)
- **LLK base** (shared): `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h` (contains `_calculate_sfpu_binary_` with DIV case and `_sfpu_binary_init_`)

There are two code paths for DIV in the SFPU:

1. **`calculate_sfpu_binary_div`** (arch-specific, used for floating-point DIV via `div_binary_tile`) -- specialized division with edge-case handling
2. **`_calculate_sfpu_binary_<..., BinaryOp::DIV>`** (LLK base, generic binary) -- simpler division without edge-case handling

The program factory routes to `calculate_sfpu_binary_div` via `llk_math_eltwise_binary_sfpu_binop_div`.

#### Annotated SFPU Kernel Source (calculate_sfpu_binary_div -- primary path)

```cpp
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Each tile in DEST occupies 32 rows when accessed via SFPI (64 / SFP_DESTREG_STRIDE = 32)
    // A full 32x32 tile has 4 faces of 16x16, each face = 8 iterations of vector width 32
    constexpr uint dst_tile_size_sfpi = 32;

    // ITERATIONS = 8: processes 8 rows per face, called 4 times (once per face) by the params wrapper
    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row (vector of 32 elements) from LHS tile in DEST
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        // Load corresponding row from RHS tile in DEST
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        // Core division: multiply LHS by reciprocal of RHS
        // _sfpu_reciprocal_<2> uses 2 Newton-Raphson iterations for high accuracy
        sfpi::vFloat result = in0 * _sfpu_reciprocal_<2>(in1);

        // Handle division edge cases using SFPU conditional execution (predicated lanes)
        v_if(in1 == 0) {
            // 0/0 = NaN
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                // x/0 = +/-Inf (sign matches numerator)
                result = std::numeric_limits<float>::infinity();
                result = sfpi::setsgn(result, in0);  // copy sign from numerator
            }
            v_endif;
        }
        // x/x = 1.0 exactly (avoids floating-point imprecision in reciprocal)
        v_elseif(in0 == in1) { result = sfpi::vConst1; }
        v_endif;

        // When not in FP32 accumulation mode, round result to BF16 using RNE
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        // Write result back to output position in DEST
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        // Advance SFPI row pointer to next row within the face
        sfpi::dst_reg++;
    }
}
```

#### Annotated SFPU Init Source

```cpp
template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        // Initialize the reciprocal lookup table (LUT) in SFPU L-registers
        // Uses 2 Newton-Raphson iterations (non-approximate mode) for precision
        _init_sfpu_reciprocal_<false>();
    }
    // ... other BINOP cases omitted
}
```

#### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[offset]` (read) | Load a vector of 32 FP32 elements from the specified DEST register row |
| `sfpi::dst_reg[offset]` (write) | Store a vector of 32 FP32 elements to the specified DEST register row |
| `sfpi::dst_reg++` | Advance the SFPI destination register row pointer by 1 |
| `_sfpu_reciprocal_<2>(x)` | Compute 1/x using SFPU reciprocal with 2 Newton-Raphson refinement iterations |
| `v_if(cond) { ... } v_endif` | SFPU predicated conditional execution -- sets lane mask based on condition |
| `v_elseif(cond) { ... } v_endif` | SFPU predicated else-if branch |
| `v_else { ... } v_endif` | SFPU predicated else branch |
| `sfpi::setsgn(value, source)` | Copy the sign bit from `source` to `value` |
| `sfpi::vConst1` | SFPU constant register holding 1.0f |
| `sfpi::reinterpret<vUInt>(x)` | Reinterpret float bits as unsigned integer (used in BF16 rounding) |
| `float32_to_bf16_rne(x)` | Software BF16 rounding using IEEE 754 Round-to-Nearest-Even |

#### SFPU Register Usage

- **DEST registers**: Two tile slots used -- `dst_index_in0 * 32` for LHS (tile 0) and `dst_index_in1 * 32` for RHS (tile 1). Output overwrites `dst_index_out * 32` (same as LHS, tile 0).
- **SFPI row pointer** (`dst_reg`): Auto-incremented through 8 rows per face. The `_llk_math_eltwise_binary_sfpu_params_` wrapper advances through all 4 faces of the 32x32 tile via `TTI_SETRWC` instructions.
- **L-registers**: Used internally by `_sfpu_reciprocal_` for the Newton-Raphson LUT coefficients, initialized by `_init_sfpu_reciprocal_<false>()`.
- **Condition codes**: Used by `v_if`/`v_elseif`/`v_else` for per-lane predication.

#### SFPU Execution Flow

1. **Initialization**: `div_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::DIV>()`, which:
   - Calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` to configure address modes and reset counters
   - Calls `_sfpu_binary_init_<APPROX, BinaryOp::DIV>()` to initialize the reciprocal LUT

2. **Per-tile execution**: `div_binary_tile(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_binop_div<APPROX, BinaryOp::DIV, DST_ACCUM_MODE>(idst0, idst1, odst)`, which:
   - Calls `_llk_math_eltwise_binary_sfpu_params_` with `calculate_sfpu_binary_div` as the callable
   - `_llk_math_eltwise_binary_sfpu_params_` sets the DEST write address, stalls until SFPU is ready, then:
     - In **RC mode** (default): iterates 4 times (once per face), calling `calculate_sfpu_binary_div` each time, advancing DEST pointer by 16 rows between faces via `TTI_SETRWC`
     - Each call to `calculate_sfpu_binary_div` processes 8 rows (ITERATIONS=8), covering one 16x16 face
   - After all 4 faces: clears the DEST register address

3. **Division math per row**: For each of the 8 rows per face (32 elements per row):
   - Load `in0` from DEST[idst0] and `in1` from DEST[idst1]
   - Compute `result = in0 * reciprocal(in1)` using Newton-Raphson reciprocal
   - Handle edge cases: 0/0 -> NaN, x/0 -> signed Inf, x/x -> 1.0
   - Optionally round to BF16 if not in FP32 accumulation mode
   - Store result to DEST[odst]

#### SFPU Configuration

- **`fp32_dest_acc_en`**: Enabled when output format is FLOAT32/INT32/UINT32, or when both inputs are FLOAT32/INT32/UINT32. Controls whether BF16 rounding is applied post-division.
- **`UnpackToDestMode::UnpackToDestFp32`**: Set for all source CBs (c_0, c_1, c_3, c_4) when the operation is SFPU and not POWER. This ensures data is unpacked to full FP32 precision in DEST.
- **`APPROX`**: Template parameter controlling approximation mode. When true, `_sfpu_reciprocal_<0>` uses 0 Newton-Raphson iterations (fast but less accurate); when false, `_sfpu_reciprocal_<2>` uses 2 iterations (more accurate).
- **Reciprocal LUT**: Initialized by `_init_sfpu_reciprocal_<false>()` during `div_binary_tile_init()`. Stores initial reciprocal approximation coefficients in SFPU L-registers.

#### Hardware Compatibility Notes
The `calculate_sfpu_binary_div` implementation is **identical** between Wormhole B0 and Blackhole architectures. Both use the same reciprocal approximation algorithm, the same edge-case handling, and the same BF16 rounding logic. The underlying `_sfpu_reciprocal_` intrinsic is provided by the SFPI library which abstracts hardware differences.

The LLK base layer (`tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`) contains a simpler generic `_calculate_sfpu_binary_` template with a `BinaryOp::DIV` case that only does `in0 * _sfpu_reciprocal_<2>(in1)` without edge-case handling. The arch-specific `calculate_sfpu_binary_div` adds the 0/0, x/0, and x/x special cases.

## Implementation Notes

1. **Non-SFPU DIV fallback**: When `is_sfpu` is false (e.g., for certain dtype combinations), DIV is decomposed into `process_rhs = RECIP` + `FpuBinaryOp::MUL`. This uses the FPU matrix multiply pipeline with reciprocal preprocessing, which may be faster for BFLOAT16 but less flexible.

2. **INT32 division**: When both inputs are INT32, a separate SFPU path is used (`div_int32_tile` / `div_int32_tile_init`), defined in `api/compute/div_int32_sfpu.h`. This uses integer-specific SFPU operations rather than the floating-point reciprocal approach.

3. **Integer division typecast bypass**: Line 600-609 of the program factory shows that typecast post-processing is explicitly skipped for integer division (`is_integer_division`), since the operation natively produces INT32 output.

4. **Broadcast optimization**: The `SubtileBroadcastType` enum controls how broadcasting is handled at the kernel level. For scalar broadcasts, the broadcast tile is loaded once and reused for all output tiles. For column broadcasts, the broadcast tile is reused for all tiles in a row (Wt tiles). This avoids redundant DRAM reads.

5. **Edge-case precision**: The `v_elseif(in0 == in1) { result = vConst1; }` check ensures that `x/x` produces exactly 1.0, avoiding the slight imprecision that `x * reciprocal(x)` would introduce. Similarly, the `0/0 -> NaN` and `x/0 -> Inf` checks match IEEE 754 semantics.

6. **BF16 rounding**: The `float32_to_bf16_rne` function implements software Round-to-Nearest-Even for BF16 conversion, ensuring correct tie-breaking behavior. This is only applied when `!is_fp32_dest_acc_en`, i.e., when the output is BF16.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary_ng program factory work? What are the different subtypes (scalar, bcast, etc.) and how does it set up kernels, circular buffers, and core distribution?"
   **Reason**: Initial architectural understanding of the binary_ng framework
   **Key Findings**: Learned about SubtileBroadcastType enum, BinaryNgKernelConfig structure, kernel selection logic, and how sharding interacts with core distribution.

2. **Query**: "What are the binary_ng compute kernels and how do they work? Specifically the eltwise_binary and eltwise_binary_sfpu kernels."
   **Reason**: Understanding the compute kernel dispatch pattern and SFPU operation mapping
   **Key Findings**: Learned that SFPU kernels use `BINARY_SFPU_OP` and `BINARY_SFPU_INIT` macros, that DIV maps to `SfpuBinaryOp::DIV`, and that the LLK layer provides `llk_math_eltwise_binary_sfpu_binop.h` with `calculate_sfpu_binary_div`.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` (lines 139-145, 394-399)
   **Reason**: Understanding how DIV is configured in OpConfig
   **Key Information**: SFPU DIV maps directly to `SfpuBinaryOp::DIV`; non-SFPU DIV decomposes to `RECIP + MUL`. The init/func pair is `("div_binary_tile_init();", "div_binary_tile")` for float and `("div_int32_tile_init();", "div_int32_tile")` for INT32.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Understanding how the SFPU function is dispatched across tile faces
   **Key Information**: `_llk_math_eltwise_binary_sfpu_params_` iterates over 4 faces in RC mode, calling the SFPU function once per face (8 iterations each), advancing DEST pointer via `TTI_SETRWC` between faces.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary.h`
   **Reason**: Understanding the base LLK SFPU binary implementation and init
   **Key Information**: `_sfpu_binary_init_` for DIV initializes the reciprocal LUT via `_init_sfpu_reciprocal_<false>()`. The generic `_calculate_sfpu_binary_` has a simpler DIV path without edge-case handling.
