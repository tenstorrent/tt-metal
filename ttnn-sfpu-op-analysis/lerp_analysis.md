# LERP (Linear Interpolation) Implementation Analysis

## Overview

LERP computes element-wise linear interpolation: `out = input + weight * (end - input)`. It is a ternary SFPU operation taking three inputs (input/start, end, weight) and producing one output. The operation supports three variant modes: TTT (tensor-tensor-tensor), TTS (tensor-tensor-scalar where weight is scalar), and TST (tensor-scalar-tensor where end is scalar). It also supports multiple broadcast types (none, column, row, scalar, outer) for each variant.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_program_factory.cpp`

## Work Unit Definition

One work unit is a single 32x32 tile. The operation processes tiles one at a time (`num_tiles_per_cycle = 1`). Each cycle reads one tile from each input, applies the SFPU lerp computation, and writes one output tile.

## Tensor Format and Layout

### Input Tensors

| Property | Input 0 (predicate/start) | Input 1 (end) | Input 2 (weight) |
|---|---|---|---|
| Dimension Convention | NCHW (up to 5D+, dims > 5 collapsed) | NCHW | NCHW |
| Tensor Layout | TILE (32x32) | TILE (32x32) | TILE (32x32) or Scalar |
| Memory Layout | Interleaved or Sharded (Height/Width/Block) | Interleaved or Sharded | Interleaved or Sharded |
| Buffer Type | DRAM or L1 | DRAM or L1 | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32 | BFLOAT16, FLOAT32 | BFLOAT16, FLOAT32 |

For LERP, the naming convention from the generic ternary framework maps as:
- "predicate" (CB0) = **input/start** tensor
- "value_true" (CB1) = **end** tensor
- "value_false" (CB2) = **weight** tensor (for TTT variant)

In TTS variant, weight is a scalar runtime argument and CB2 is not used. TST variant is theoretically supported (end is scalar), though less common for lerp.

### Output Tensor

| Property | Output |
|---|---|
| Dimension Convention | NCHW (broadcasted shape of inputs) |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Interleaved or Sharded |
| Buffer Type | DRAM or L1 |
| Data Type | BFLOAT16, FLOAT32 |

### Layout Transformations

No tilize/untilize is performed. All inputs and outputs must already be in TILE layout. Broadcasting is handled at the sub-tile level in the compute kernel (column broadcast via SFPI fill) or at the reader level (row/outer broadcast via tile replication). For non-BFLOAT16 row broadcast, the reader fills entire tiles; for BFLOAT16 row broadcast, the LLK `unary_bcast<ROW>` is used in the compute kernel.

## Data Flow Pattern

### TTT No-Broadcast Path (simplest case)
1. **Reader** reads one tile each from input (CB0), end (CB1), weight (CB2) from DRAM/L1 via NoC, pushes to respective CBs
2. **Compute** waits for all three CBs, copies tiles to DEST registers 0/1/2 via `copy_tile`, runs SFPU `lerp_tile`, packs result to CB3
3. **Writer** waits for CB3, writes output tile to DRAM/L1 via NoC

### TTT Column/Scalar Broadcast Path
1. **Reader** reads tiles accounting for broadcast (broadcast tensors read once, non-broadcast tensors read per-tile)
2. **Compute** uses `process_tile()` loop with `tile_freq`/`tile_start` to repeat broadcast tiles across the broadcast dimension. Broadcast CBs are waited/popped outside the inner loop; non-broadcast CBs are waited/popped inside.
3. **Writer** writes output tiles sequentially

### TTT Row Broadcast Path (BFLOAT16)
1. **Reader** reads row-broadcast tiles (single row of tiles) into pre-CBs (CB0-CB2)
2. **Compute** uses `unary_bcast<ROW>` LLK to expand row tiles into full tiles in bcast CBs (CB4-CB6), then runs ternary SFPU on the effective CBs
3. **Writer** writes output tiles

### TTS/TST Path
1. **Reader** reads two tensors into CB0 and CB1
2. **Compute** copies tensor tiles to DEST, uses `fill_tile` to fill scalar value into the remaining DEST register, then runs SFPU `lerp_tile`
3. **Writer** writes output

### Sharded Path
When inputs/output are sharded in L1, the reader simply does `cb_reserve_back`/`cb_push_back` to expose the already-present L1 shard data to the compute kernel. The writer is also a no-op when the output is sharded (data stays in L1).

## Circular Buffer Configuration

### TTT Variant (No Broadcast / Outer Broadcast)

| CB ID | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|
| c_0 | Input/start tensor | 2 (or shard volume if sharded) | 1 | Double-buffered | Reader | Compute |
| c_1 | End tensor | 2 (or shard volume if sharded) | 1 | Double-buffered | Reader | Compute |
| c_2 | Weight tensor | 2 (or shard volume if sharded) | 1 | Double-buffered | Reader | Compute |
| c_3 | Output tensor | 2 (or shard volume if sharded) | 1 | Double-buffered | Compute | Writer |

### TTT Variant (Row Broadcast, BFLOAT16)
All of the above plus:

| CB ID | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|
| c_4 | Broadcast-expanded input | 2 | 1 | Double-buffered | Compute (bcast stage) | Compute (SFPU stage) |
| c_5 | Broadcast-expanded end | 2 | 1 | Double-buffered | Compute (bcast stage) | Compute (SFPU stage) |
| c_6 | Broadcast-expanded weight | 2 | 1 | Double-buffered | Compute (bcast stage) | Compute (SFPU stage) |

### TTS/TST Variant

| CB ID | Purpose | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|
| c_0 | Input/start tensor | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute |
| c_1 | End tensor (TTS) or Weight tensor (TST) | 2 (or shard volume) | 1 | Double-buffered | Reader | Compute |
| c_3 | Output tensor | 2 (or shard volume) | 1 | Double-buffered | Compute | Writer |

## Pipeline Pattern Summary

All CBs are allocated with capacity 2 and consumed 1 tile at a time, enabling double-buffering. This allows the reader to prefetch the next tile while the compute kernel processes the current one, and compute to produce the next output while the writer drains the current one.

For sharded tensors, the CB capacity equals the shard volume (all tiles in the shard), so the entire shard is available at once -- no pipelining needed since data is already in L1.

## Index Calculations

The reader kernel uses a multi-dimensional index decomposition to map a linear tile ID to per-tensor tile offsets. For each tensor, strides are computed as:
- `nD_stride = Ht * Wt * C * N * D * (ND > 1)`
- `d_stride = Ht * Wt * C * N * (D > 1)`
- `n_stride = Ht * Wt * C * (N > 1)`
- `c_stride = Ht * Wt * (C > 1)`

The stride being zero when a dimension is 1 implements broadcasting along that dimension.

For width-sharded tensors, the reader limits tile iteration to `start_tw + dst_shard_width` rather than the full `Wt`, and the writer adjusts the output offset to skip the non-shard portion of each row.

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads within each (nd, d, n, c, h) row, with stride jumps between rows and higher dimensions. Each tile read is a NoC async read followed by a barrier.
- **Sharded**: No reads needed; data already in L1. Reader just exposes the shard via `cb_reserve_back`/`cb_push_back`.

### Write Pattern
- **Interleaved**: Sequential tile writes mirroring the reader's iteration order. Each tile is written via NoC async write with barrier.
- **Sharded**: No writes needed; output remains in L1.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Rectangular (prefers zero-start grid at (0,0)) |
| Work Splitting | `split_work_to_cores()` divides total output tiles across available cores |
| Core Group 1 | Gets `num_tiles_per_core_group_1` tiles |
| Core Group 2 | Gets `num_tiles_per_core_group_2` tiles (may be 0 for remainder cores) |
| Load Balancing | Even split with remainder distributed to core_group_2 |
| Remainder Handling | Cores not in either group get zero-args (no-op) |
| Sharded | Core grid from shard spec; each core processes its shard |
| Row Major | Default true; follows shard orientation if sharded |

## Arguments

### Compile-Time Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles_per_cycle | uint32_t | Always 1; tiles processed per read-compute-write cycle |
| 1 | scalar_is_true_value | uint32_t | 0 for TTS (scalar is "false"/weight), 1 for TST (scalar is "true"/end); TTT variant ignores this |

#### Reader Kernel (TTT)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_src0 | uint32_t | CB index for input/start tensor (c_0) |
| 1 | cb_id_src1 | uint32_t | CB index for end tensor (c_1) |
| 2 | cb_id_src2 | uint32_t | CB index for weight tensor (c_2) |
| 3+ | TensorAccessorArgs (src0) | varies | Compile-time tensor accessor config for input tensor |
| ... | TensorAccessorArgs (src1) | varies | Compile-time tensor accessor config for end tensor |
| ... | TensorAccessorArgs (src2) | varies | Compile-time tensor accessor config for weight tensor |
| last | has_sharding | uint32_t | 1 if using native L1 sharding, 0 otherwise |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | CB index for output tensor (c_3) |
| 1+ | TensorAccessorArgs (dst) | varies | Compile-time tensor accessor config for output |
| last | has_sharding | uint32_t | 1 if output is sharded, 0 otherwise |

### Runtime Arguments

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | num_tiles | uint32_t | Number of tiles this core must process |
| 1 | freq | uint32_t | Broadcast frequency (tiles before broadcast tile repeats); 0 for no-broadcast/row-broadcast |
| 2 | counter | uint32_t | Starting tile offset within broadcast cycle |
| 3 | scalar_arg | uint32_t | Packed scalar value (bit-cast float/int); 0 for TTT variant |

#### Reader Kernel (TTT)

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | DRAM/L1 address of input/start tensor |
| 1 | src1_addr | uint32_t | DRAM/L1 address of end tensor |
| 2 | src2_addr | uint32_t | DRAM/L1 address of weight tensor |
| 3 | num_tiles | uint32_t | Tiles per core |
| 4 | start_id | uint32_t | Starting tile ID for this core |
| 5-8 | pred strides | uint32_t | nD/d/n/c strides for input tensor |
| 9-14 | output dims | uint32_t | D, N, C, Ht, Wt, ND of output shape |
| 15-19 | true strides + count | uint32_t | nD/d/n/c strides and tile count for end tensor |
| 20-24 | false strides + count | uint32_t | nD/d/n/c strides and tile count for weight tensor |
| 25 | dst_shard_width | uint32_t | Width of output shard in tiles (0 if not sharded) |
| 26 | src_num_tiles | uint32_t | Total tiles in input shard (0 if not sharded) |

#### Writer Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | DRAM/L1 address of output buffer |
| 1 | num_tiles | uint32_t | Tiles this core must write |
| 2 | start_id | uint32_t | Starting tile ID |
| 3 | dst_shard_width | uint32_t | Shard width in tiles |
| 4-9 | output dims | uint32_t | D, N, C, Ht, Wt, ND |
| 10 | padding | uint32_t | Reserved (0) |

## Kernel Implementations

### Reader Kernel (TTT No-Broadcast/Outer-Broadcast)

- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nosubtilebcast_ttt.cpp`
- **Key Logic**: Iterates through ND, D, N, C, H, W dimensions using stride-based offsets for each of the three tensors. Handles both interleaved (NoC reads) and sharded (L1 expose) paths via `SRC_SHARDED_A/B/C` defines. Broadcasting along outer dimensions is achieved through zero strides -- when a dimension is 1 for a tensor, its stride is 0, so the tile offset does not advance.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_writer_nobcast.cpp`
- **Key Logic**: Mirrors reader iteration order. For interleaved outputs, writes one tile at a time with NoC async write + barrier. For sharded outputs (`DST_SHARDED` define), the entire kernel body is compiled out -- output stays in L1.

### Compute Kernel

The LERP operation uses one of five compute kernel files depending on variant and broadcast type:

| Variant | Broadcast | Compute Kernel File |
|---|---|---|
| TTT | None, Outer | `ternary_sfpu_no_bcast_ttt.cpp` |
| TTT | Column, Scalar | `ternary_sfpu_col_scalar_bcast_ttt.cpp` |
| TTT | Row (BF16) | `ternary_sfpu_row_bcast_ttt.cpp` |
| TTS/TST | None, Outer, Row | `ternary_sfpu_no_bcast_tts_tst.cpp` |
| TTS/TST | Column, Scalar | `ternary_sfpu_col_scalar_bcast_tts_tst.cpp` |

All kernels share the same SFPU dispatch pattern: they load tiles to DEST[0], DEST[1], DEST[2], call `TERNARY_SFPU_OP_INIT()` and `TERNARY_SFPU_OP_FUNC(0, 1, 2, 0)`, then pack DEST[0] to the output CB. The difference is only in how tiles are acquired (direct vs broadcast-aware synchronization) and whether scalars are filled via `fill_tile`.

#### Compute Kernel File (Primary: TTT No-Broadcast)

`ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_no_bcast_ttt.cpp`

#### Annotated Compute Kernel Source (TTT No-Broadcast)

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"     // provides lerp_tile() and lerp_tile_init()

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);  // runtime arg 0: total tiles for this core

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // always 1

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;  // CB for input/start tensor
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;  // CB for end tensor
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;  // CB for weight tensor
    constexpr auto cb_out = tt::CBIndex::c_3;       // CB for output tensor

    unary_op_init_common(cb_pre_in1, cb_out);  // initializes unpack/pack pipeline for SFPU mode

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Wait for reader to produce one tile in each input CB
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);  // wait for input/start tile
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);  // wait for end tile
        cb_wait_front(cb_pre_in3, num_tiles_per_cycle);  // wait for weight tile

        cb_reserve_back(cb_out, num_tiles_per_cycle);  // reserve space in output CB for writer

        tile_regs_acquire();  // acquire DEST register file for exclusive use

        // Unpack (copy) tiles from CBs into DEST registers via the unpack pipeline
        copy_tile_to_dst_init_short(cb_pre_in1);  // configure unpacker for cb_pre_in1 format
        copy_tile(cb_pre_in1, 0, 0);              // copy tile 0 from cb_pre_in1 -> DEST[0] (input/start)

        copy_tile_to_dst_init_short(cb_pre_in2);  // reconfigure unpacker for cb_pre_in2 format
        copy_tile(cb_pre_in2, 0, 1);              // copy tile 0 from cb_pre_in2 -> DEST[1] (end)

        copy_tile_to_dst_init_short(cb_pre_in3);  // reconfigure unpacker for cb_pre_in3 format
        copy_tile(cb_pre_in3, 0, 2);              // copy tile 0 from cb_pre_in3 -> DEST[2] (weight)

        // For LERP: expands to lerp_tile_init() then lerp_tile<format>(0, 1, 2, 0)
        TERNARY_SFPU_OP_INIT();           // configures SFPU pipeline for lerp operation
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0); // SFPU lerp: DEST[0] = DEST[0] + DEST[2] * (DEST[1] - DEST[0])

        tile_regs_commit();  // signal that DEST writes are complete, hand off to packer
        tile_regs_wait();    // wait for packer to be ready to read DEST

        pack_tile(0, cb_out);  // pack DEST[0] (result) into output CB

        tile_regs_release();  // release DEST registers for next iteration

        cb_push_back(cb_out, num_tiles_per_cycle);    // publish output tile to writer
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle); // free consumed input/start tile
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle); // free consumed end tile
        cb_pop_front(cb_pre_in3, num_tiles_per_cycle); // free consumed weight tile
    }
}
```

### SFPU Kernel Implementation

#### SFPU Kernel File

`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h`
(Identical for Blackhole: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h`)

#### Annotated SFPU Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"                   // SFPI programming interface: vFloat, dst_reg, etc.
#include "ckernel_sfpu_binary.h"    // shared binary SFPU utilities

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const uint dst_index_in0,  // DEST register index for input/start tile
    const uint dst_index_in1,  // DEST register index for end tile
    const uint dst_index_in2,  // DEST register index for weight tile
    const uint dst_index_out) {  // DEST register index for output tile (same as in0 typically)
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    // Each tile in DEST has 32 rows when accessed via SFPI (64 / SFP_DESTREG_STRIDE = 32)
    // A 32x32 tile has 1024 elements = 4 faces * 16 rows * 16 cols
    // SFPI processes one row (16 elements) per iteration via SIMD vector operations
    // With ITERATIONS=8, the outer loop runs 8 times; the #pragma unroll 8 expands it
    // This covers 8 rows per face, and the _llk_math_eltwise_ternary_sfpu_params_ wrapper
    // calls this function multiple times for different faces (RC mode = all 4 faces)
    constexpr uint dst_tile_size_sfpi = 32;

    // lerp formula: out = input + weight * (end - input)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // Load one row (16 elements) from each input tile's current face row
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];  // input/start value
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];  // end value
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];  // weight value

        // Compute lerp: input + weight * (end - input)
        // This uses SFPU vector float arithmetic: subtract, multiply, add
        // Each operation processes 16 elements in parallel (one SFPI row)
        sfpi::vFloat result = in0 + in2 * (in1 - in0);

        // When not using FP32 DEST accumulation, round result to BF16
        // This prevents precision loss when the DEST register is in BF16 mode
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);  // round-to-nearest-even BF16 conversion
        }

        // Write result back to the output DEST register at the current row
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;

        // Advance the SFPI row pointer to the next row within the face
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|---|---|
| `sfpi::dst_reg[idx]` (load) | Loads a row of 16 float elements from the DEST register file at the given offset into an SFPI `vFloat` vector register |
| `sfpi::dst_reg[idx]` (store) | Stores a `vFloat` vector register back to the DEST register file at the given offset |
| `sfpi::dst_reg++` | Advances the SFPI base row pointer by 1, moving to the next row within the tile face |
| `operator-` (vFloat) | SFPU vector subtract: computes element-wise `in1 - in0` (16 elements) |
| `operator*` (vFloat) | SFPU vector multiply: computes element-wise `in2 * (in1 - in0)` (16 elements) |
| `operator+` (vFloat) | SFPU vector add: computes element-wise `in0 + product` (16 elements) |
| `float32_to_bf16_rne` | Converts FP32 result to BF16 using round-to-nearest-even; used when DEST is not in FP32 accumulation mode |

#### SFPU Register Usage

- **DEST[0]** (dst_index_in0): Holds the input/start tile. Also serves as the output destination (dst_index_out = 0), so results overwrite the input tile in-place.
- **DEST[1]** (dst_index_in1): Holds the end tile. Read-only during SFPU execution.
- **DEST[2]** (dst_index_in2): Holds the weight tile. Read-only during SFPU execution.
- **vFloat registers**: Three temporaries (`in0`, `in1`, `in2`) and one result (`result`) used within each SFPI iteration. These are SFPU-internal vector registers holding 16 float elements each.

#### SFPU Execution Flow

1. **Tile Acquisition**: The compute kernel calls `cb_wait_front` on CB0, CB1, CB2 to ensure reader has produced tiles.
2. **Unpack to DEST**: `copy_tile` is called three times to unpack tiles from CBs into DEST[0] (input), DEST[1] (end), DEST[2] (weight). Each `copy_tile_to_dst_init_short` reconfigures the unpacker for the source CB's data format.
3. **SFPU Init**: `lerp_tile_init()` calls `llk_math_eltwise_ternary_sfpu_lerp_init<APPROX>()` which calls `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::lerp>()` to configure the SFPU pipeline.
4. **SFPU Math**: `lerp_tile<format>(0, 1, 2, 0)` dispatches through the LLK layer:
   - `_llk_math_eltwise_ternary_sfpu_params_` is called with `calculate_lerp` as the function pointer
   - It calls `_llk_math_eltwise_ternary_sfpu_start_` to set up the SFPU
   - In RC (all faces) mode, `calculate_lerp` is called 4 times (once per face), each time processing 8 rows of 16 elements = 128 elements per face, totaling 512 elements per tile half. With 2 half-tiles this covers the full 1024-element tile.
   - Each invocation of `calculate_lerp` loops 8 times (ITERATIONS=8), processing one row of 16 elements per iteration via SFPI SIMD vector operations
   - `_llk_math_eltwise_ternary_sfpu_done_` finalizes the SFPU
5. **Pack**: `pack_tile(0, cb_out)` packs DEST[0] (which now holds the lerp result) into the output CB.
6. **Tile Release**: `cb_push_back` publishes the output; `cb_pop_front` frees input tiles.

#### SFPU Configuration

- **TERNARY_SFPU_OP_INIT** define: Resolves to `lerp_tile_init` for LERP op type
- **TERNARY_SFPU_OP_FUNC** define: Resolves to `lerp_tile<DataFormat::Float32>` or `lerp_tile<DataFormat::Float16_b>` depending on the input tensor's data type
- **APPROX**: Template parameter inherited from the compute config; controls approximation mode (not used by lerp since it only uses basic arithmetic)
- **DST_ACCUM_MODE**: Set to `is_fp32_dest_acc_en` based on output data format. When true, DEST registers use FP32 accumulation; when false, BF16 with explicit `float32_to_bf16_rne` rounding
- **Math Fidelity**: Not directly applicable to lerp (no LUT operations); the operation uses only basic SFPU arithmetic (+, -, *)
- **UnpackToDestMode**: Set per-CB based on tensor dtype. Float32 tensors use `UnpackToDestFp32`; others use `Default`

#### Hardware Compatibility Notes

The SFPU lerp kernel implementation is **identical** between Wormhole B0 and Blackhole architectures. Both `ckernel_sfpu_lerp.h` files contain the same `calculate_lerp` function. The LLK wrapper `llk_math_eltwise_ternary_sfpu_lerp.h` is also identical across architectures. The only architecture-specific component is the underlying `_llk_math_eltwise_ternary_sfpu_params_` function (in the tt_llk submodule), which handles face iteration and SFPU pipeline setup -- but its interface and behavior are the same.

INT32 and UINT32 data types are **not supported** for LERP (unlike WHERE which supports them). The `get_compute_defines` function only generates Float32 and Float16_b format templates for LERP.

## Implementation Notes

1. **Shared Infrastructure with WHERE**: LERP reuses the same program factory, compute kernels, reader/writer kernels, and CB configuration as the WHERE operation. The only difference is the `TERNARY_SFPU_OP_INIT` and `TERNARY_SFPU_OP_FUNC` defines, which are set to `lerp_tile_init`/`lerp_tile` instead of `where_tile_init`/`where_tile`.

2. **In-place Output**: The SFPU result is written to DEST[0] (same register as input/start), meaning the lerp output overwrites the input tile. This is efficient since input is consumed and no longer needed.

3. **Broadcast Handling**: The operation supports complex broadcast patterns including column broadcast (weight dimension = 1), row broadcast (height dimension = 1), scalar broadcast (both H and W = 1), and outer dimension broadcast (only batch/channel dimensions differ). Each pattern uses a different reader and compute kernel pair.

4. **No FPU Path for LERP**: Unlike ADDCMUL/ADDCDIV which have FPU variants for matching-dtype BFLOAT16 inputs, LERP always uses the SFPU path. The `is_fpu` flag is only set for ADDCMUL/ADDCDIV operations.

5. **TTS Variant for Common Use Case**: When LERP is called with a scalar weight (the most common PyTorch usage pattern), it uses the TTS variant which avoids allocating and reading a third tensor, instead using `fill_tile` to broadcast the scalar into DEST[2].

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the ternary eltwise program factory work for operations like lerp? What kernels does it use and how does it distribute work across cores?"
   **Reason**: Initial reconnaissance to understand the program factory structure and kernel selection mechanism.
   **Key Findings**: Confirmed that LERP uses `TernaryKernelConfig` for kernel selection based on op type, variant, and broadcast type. Work distribution uses `split_work_to_cores()`. Shared infrastructure with WHERE.

2. **Query**: "Where is the function `_llk_math_eltwise_ternary_sfpu_params_` defined? What does it do?"
   **Reason**: Needed to understand the LLK wrapper that iterates over tile faces when dispatching the SFPU kernel.
   **Key Findings**: Defined in `tt_llk` submodule under `llk_lib/llk_math_eltwise_ternary_sfpu_params.h`. It handles face iteration (RC mode = all 4 faces), SFPU start/done lifecycle, and dispatches the provided sfpu_func for each face.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp`
   **Reason**: Needed to understand kernel file path mapping and compute defines for LERP.
   **Key Information**: LERP maps to `lerp_tile_init`/`lerp_tile<format>` defines. Kernel config map shows LERP uses same kernel files as WHERE.

2. **Source**: `ttnn/cpp/ttnn/operations/eltwise/ternary/common/ternary_op_types.hpp`
   **Reason**: Needed enum definitions for op types, variants, and broadcast types.
   **Key Information**: LERP = linear interpolation `out = input + weight * (end - input)`. Supports TTT, TTS, TST variants.
