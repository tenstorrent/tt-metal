# Binary Left Shift Implementation Analysis

## Overview

The binary left shift operation computes element-wise left bit-shift: `y = x0 << x1`, where `x0` is the value tensor and `x1` is the shift amount tensor. Both inputs must be integer types (Int32, UInt32, or UInt16). The operation is dispatched through the SFPU (Scalar Floating Point Unit) because the FPU matrix engine does not support integer bitwise shift operations.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

One work unit is a **tile** (32x32 elements). The compute kernel processes tiles in blocks of `per_core_block_size` tiles per iteration, iterating `per_core_block_cnt` times. In the non-sharded interleaved case, `block_size = 1` and `block_cnt = num_tiles_per_core`, so each work unit is a single tile. In the sharded case, `block_size = find_max_block_size(num_tiles_per_shard)` and `block_cnt = num_tiles_per_shard / block_size`.

## Tensor Format and Layout

### Input Tensor(s)

| Property | Input A (value) | Input B (shift amount) |
|---|---|---|
| **Rank** | Arbitrary (flattened to tiles) | Same shape as A |
| **Dimension Convention** | NHWC/Row-major | Same as A |
| **Tensor Layout** | TILE (32x32) | TILE (32x32) |
| **Memory Layout** | Interleaved or Sharded | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1 | DRAM or L1 |
| **Data Type** | Int32, UInt32, or UInt16 | Same as A |

### Output Tensor(s)

| Property | Output |
|---|---|
| **Rank** | Same as Input A |
| **Dimension Convention** | Same as Input A |
| **Tensor Layout** | TILE (32x32) |
| **Memory Layout** | Interleaved or Sharded |
| **Buffer Type** | DRAM or L1 |
| **Data Type** | Same as Input A |

### Layout Transformations

No layout transformations (tilize/untilize) occur within the program factory. Data is expected to be in tiled format on entry and remains tiled on output.

## Data Flow Pattern

1. **Reader kernel** reads tiles from input tensor A into CB c_0 and from input tensor B into CB c_1. If inputs are sharded, the reader simply marks the sharded buffer as available (reserve + push_back) without NoC reads.
2. **Compute kernel** waits for tiles in CB c_0 (input A) and CB c_1 (input B). It copies input A tiles into even DEST register slots (`i*2`) and input B tiles into odd DEST register slots (`i*2+1`). The SFPU then executes `binary_left_shift_tile` which reads from DEST[i*2] and DEST[i*2+1], computes `x0 << x1`, and writes the result back to DEST[i*2]. The result is packed from DEST[i*2] into CB c_2.
3. **Writer kernel** reads tiles from CB c_2 and writes them to the output buffer via NoC. If the output is sharded, it simply waits for all tiles to appear in the output CB (which is backed by the output buffer directly).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Data Format | Capacity (tiles) | Block Size (tiles) | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| c_0 | cb_src0 | Input A tiles | src0 dtype | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (interleaved) or `max_block_size` (sharded) | Double-buffered (interleaved) or single-buffered (sharded) | Reader | Compute |
| c_1 | cb_src1 | Input B tiles | src1 dtype | Sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (interleaved) or `max_block_size` (sharded) | Double-buffered (interleaved) or single-buffered (sharded) | Reader | Compute |
| c_2 | cb_out0 | Output tiles | output dtype | Sharded/block_or_width_sharded: `num_tiles_per_shard`; Interleaved: `2 * max_block_size` | 1 (interleaved) or `max_block_size` (sharded) | Double-buffered (interleaved) or single-buffered (sharded) | Compute | Writer |

**Note**: CB c_3 and c_4 (interim buffers for pre-scaling input A and B respectively) are NOT allocated for binary left shift because `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` are not defined for this operation.

## Pipeline Pattern Summary

- **Interleaved path**: CBs c_0, c_1, and c_2 each have capacity `2 * max_block_size` with a block size of `max_block_size` (where `max_block_size = 1` for interleaved). This gives 2x capacity vs. block size, enabling **double-buffering** -- the reader can fill the next block while compute processes the current one.
- **Sharded path**: CBs are sized to hold the entire shard (`num_tiles_per_shard` tiles). All data is available at once; no pipelining is needed since the data is already in L1.

## Index Calculations

- **Interleaved**: Tiles are accessed linearly. Each core starts from `start_id` (a global tile offset) and processes `num_tiles_per_core` consecutive tiles. The `TensorAccessor` maps tile IDs to physical DRAM/L1 addresses using bank mapping.
- **Block/Width sharded**: Tiles are addressed with `(block_height, block_width)` coordinates. The `start_id` is computed as `(core_row * block_height * block_width * num_shards_per_width) + (core_col * block_width)`. The reader iterates row by row: `row_start_tile_id += num_cores_y * block_width`.
- **Compute kernel DEST indexing**: Input A tile `i` is placed at DEST[i*2], input B tile `i` at DEST[i*2+1]. The SFPU operation reads from these two slots and writes back to DEST[i*2], which is then packed out.

## Memory Access Patterns

### Read Pattern

- **Interleaved**: Sequential tile-by-tile reads. For each tile, the reader issues `noc_async_read_tile` for both src0 and src1 in lockstep, then waits for completion (`noc_async_read_barrier`) before pushing both tiles.
- **Sharded**: No read operations -- the CB is directly backed by the sharded tensor's L1 buffer. The reader just marks all tiles as available.

### Write Pattern

- **Interleaved**: Sequential tile-by-tile writes. The writer reads one tile from CB c_2, issues `noc_async_write_page`, flushes, and pops. A final `noc_async_write_barrier` at the end ensures all writes complete.
- **Sharded**: No write operations -- the output CB is directly backed by the output tensor's L1 buffer. The writer just waits for all tiles to appear.

## Core Distribution Strategy

| Property | Value |
|---|---|
| **Grid Topology** | Determined by `operation_attributes.worker_grid` (typically full device grid or shard grid) |
| **Work Splitting** | `split_work_to_cores` for interleaved; shard grid for sharded |
| **Load Balancing** | Two core groups: group 1 gets `num_tiles_per_core_group_1` tiles, group 2 gets `num_tiles_per_core_group_2` (one fewer). Sharded uses a single core group. |
| **Remainder Handling** | Extra tiles distributed to group 1 cores; group 2 gets one fewer tile. Non-working cores get zero tile counts. |
| **Core Ordering** | Row-major (interleaved) or shard orientation (sharded) |

## Arguments

### Compile-Time Arguments

**Reader kernel** (`reader_binary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | block_or_width_sharded | uint32_t | 1 if block or width sharded, 0 otherwise |
| 1+ | TensorAccessorArgs (src0) | multiple | Bank mapping args for input A (only if not IN0_SHARDED) |
| N+ | TensorAccessorArgs (src1) | multiple | Bank mapping args for input B (only if not IN1_SHARDED) |

**Writer kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | cb_id_out | uint32_t | Output CB index (c_2) |
| 1+ | TensorAccessorArgs (dst) | multiple | Bank mapping args for output buffer |

**Compute kernel** (`eltwise_binary_sfpu_kernel.cpp`):

Compile-time arguments are passed via `#define` macros rather than `get_compile_time_arg_val()`:

| Define | Value for LEFT_SHIFT | Description |
|---|---|---|
| `SHIFT_INIT` | `binary_shift_tile_init();` | Initializes SFPU for shift operations |
| `BINARY_SFPU_OP` | `binary_left_shift_tile<DataFormat::XXX>(i*2, i*2+1, i*2);` | Executes left shift on tile pair; XXX is Int32/UInt32/UInt16 based on input dtypes |

### Runtime Arguments

**Reader kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src0_addr | uint32_t | Base address of input tensor A |
| 1 | src1_addr | uint32_t | Base address of input tensor B |
| 2 | num_tiles | uint32_t | Number of tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 if interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 if interleaved) |
| 6 | num_cores_y | uint32_t | Number of shards per width (0 if interleaved) |

**Compute kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

**Writer kernel**:

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Base address of output buffer |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

## Kernel Implementations

### Reader Kernel

| Property | Value |
|---|---|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| **Type** | ReaderDataMovement |
| **I/O** | Reads: DRAM/L1 input buffers; Writes: CB c_0, CB c_1 |
| **Sync** | `cb_reserve_back` / `cb_push_back` per tile; `noc_async_read_barrier` between NoC read and push |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: The kernel has three code paths selected by `#define` macros: (1) both inputs sharded -- just mark all tiles as available, (2) block/width sharded -- 2D tile iteration with `row_start_tile_id` advancing by `num_cores_y * block_width`, (3) interleaved -- simple linear tile loop. Uses `TensorAccessor` for NoC address resolution.

### Writer Kernel

| Property | Value |
|---|---|
| **File** | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| **Type** | WriterDataMovement |
| **I/O** | Reads: CB c_2; Writes: DRAM/L1 output buffer |
| **Sync** | `cb_wait_front` / `cb_pop_front` per tile; `noc_async_writes_flushed` per tile; `noc_async_write_barrier` at end |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: For sharded output, simply `cb_wait_front` for all tiles (the CB is backed by the output buffer). For interleaved, iterate linearly from `start_id`, writing one tile at a time using `noc_async_write_page` via `TensorAccessor`.

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
#include "api/compute/binary_shift.h"         // provides binary_left_shift_tile, binary_shift_tile_init
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

// PRE_SCALE is true if either input requires pre-processing (e.g., activation on input A).
// For binary left shift, neither SFPU_OP_INIT_PRE_IN0_0 nor SFPU_OP_INIT_PRE_IN1_0 is defined,
// so PRE_SCALE is false and the pre-scaling blocks are compiled out.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // number of blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);   // number of tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // CB for input A (value)
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // CB for input B (shift amount)

    // For left shift, no pre-scaling is defined, so cb_inp0 == cb_in0 and cb_inp1 == cb_in1.
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;   // interim CB for pre-processed input A
#else
    constexpr auto cb_inp0 = cb_in0;              // no pre-processing: use input A directly
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;   // interim CB for pre-processed input B
#else
    constexpr auto cb_inp1 = cb_in1;              // no pre-processing: use input B directly
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;   // output CB

    unary_op_init_common(cb_in0, cb_out0);        // initialize unpack/pack pipeline for cb_in0 -> cb_out0

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));   // not used for left shift
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // --- PRE_SCALE blocks are compiled out for left shift ---
        // (SFPU_OP_INIT_PRE_IN0_0 and SFPU_OP_INIT_PRE_IN1_0 are not defined)

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        // ... pre-processing of input A (not active for left shift)
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
        // ... pre-processing of input B (not active for left shift)
#endif

        // Wait for both inputs to be available in their respective CBs
        cb_wait_front(cb_inp0, per_core_block_size);   // blocks until reader has produced input A tiles
        cb_wait_front(cb_inp1, per_core_block_size);   // blocks until reader has produced input B tiles
        cb_reserve_back(cb_out0, per_core_block_size);  // reserve output space

        tile_regs_acquire();   // acquire DEST register file for SFPU use
        tile_regs_wait();      // wait for register file to be available

        // Copy input B tiles into odd DEST slots: DEST[i*2+1]
        // First, initialize the copy operation for input B's data format
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);   // copy input A tile i from CB c_0 to DEST[i*2]
        }

        // Switch data format config for input B
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);   // copy input B tile i from CB c_1 to DEST[i*2+1]

            // For LEFT_SHIFT, SHIFT_INIT is defined as binary_shift_tile_init()
            // This is called once per tile (inside the loop) to configure the SFPU pipeline
#ifdef SHIFT_INIT
            SHIFT_INIT   // expands to: binary_shift_tile_init();
#endif

            // BINARY_SFPU_OP expands to:
            // binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2);
            // This reads DEST[i*2] (value) and DEST[i*2+1] (shift amount),
            // computes value << shift_amount, and stores result in DEST[i*2].
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP
#endif

            // Pack the result from DEST[i*2] into the output CB
            pack_tile(i * 2, cb_out0);
        }

        tile_regs_commit();    // signal that DEST registers are done being written
        tile_regs_release();   // release DEST register file

        cb_pop_front(cb_inp0, per_core_block_size);    // free input A tiles from CB
        cb_pop_front(cb_inp1, per_core_block_size);    // free input B tiles from CB
        cb_push_back(cb_out0, per_core_block_size);    // publish output tiles to writer
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

`tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h` (Blackhole)
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_shift.h` (Wormhole B0)

The thin wrapper in tt-metal that calls into this: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_shift.h`

#### Annotated SFPU Kernel Source

**Blackhole implementation** (`tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h`):

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_binary_left_shift_(
    const std::uint32_t dst_index_in0,    // DEST tile index for input A (value)
    const std::uint32_t dst_index_in1,    // DEST tile index for input B (shift amount)
    const std::uint32_t dst_index_out)    // DEST tile index for output (same as dst_index_in0)
{
    // Validate that INSTRUCTION_MODE is one of the allowed integer modes
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE),
        "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Select load/store instruction modifier based on data format:
    // - SIGN_MAGNITUDE_FORMAT=true -> INT32_2S_COMP (two's complement interpretation)
    // - Otherwise use the INSTRUCTION_MODE directly (INT32 for Int32/UInt32, LO16 for UInt16)
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    // SFPU microcode: iterate 8 times to process all 8 rows of 4 elements each per face.
    // The outer _llk_math_eltwise_binary_sfpu_params_ function calls this 4 times (once per face
    // in RC vector mode), so total coverage is 4 faces * 8 iterations * 4 elements = 128 = 32x32/8.
    // Actually: each SFPLOAD loads a vector of 32 elements (one row of a 16x16 face).
    // 8 iterations * 4 faces = 32 rows = 1024 elements = one 32x32 tile.
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile in DEST occupies 64 row-slots (4 faces of 16 rows each)
        constexpr std::uint32_t dst_tile_size = 64;

        // Load one row-vector from DEST[dst_index_in0] into SFPU local register LREG0 (the value)
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        // Load one row-vector from DEST[dst_index_in1] into SFPU local register LREG1 (shift amount)
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);

        // --- Boundary check: if shift_amount < 0 OR shift_amount >= 32, result = 0 ---

        // SFPSETCC with mode 4: sets condition code if LREG1 (shift amount) is negative
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);

        // SFPIADD: add -32 (0xFE0 in 12-bit signed) to LREG1, store in LREG2.
        // Mode 1: conditional execution -- only runs where CC is NOT set (shift >= 0).
        // After this, LREG2 = shift_amount - 32.
        // Then sets CC if LREG2 >= 0 (i.e., shift_amount >= 32).
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);

        // SFPCOMPC: complement the condition code.
        // Now CC is set for lanes where shift_amount < 0 OR shift_amount >= 32.
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // SFPMOV with CC active: for lanes where shift is out-of-range, set LREG0 = 0 (LCONST_0)
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // SFPENCC: disable conditional execution (clear CC), return to unconditional mode
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // --- Perform the actual left shift ---
        // SFPSHFT: shift LREG0 left by the amount in LREG1.
        // SFPSHFT uses LREG1 as the shift amount (positive = left shift).
        // Result is stored in LREG0.
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

        // Store the result from LREG0 back to DEST[dst_index_out]
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);

        // Advance the DEST row pointer for the next iteration
        sfpi::dst_reg++;
    }
}
```

**Wormhole B0 implementation** is identical except it uses `ADDR_MOD_3` instead of `ADDR_MOD_7` for the `TT_SFPLOAD` and `TT_SFPSTORE` address modifiers. This is an architecture-specific difference in how the DEST register address auto-increment is configured.

#### SFPU Instructions Used

| Instruction | Description |
|---|---|
| `TT_SFPLOAD` | Loads a vector of elements from a DEST register row into an SFPU local register (LREG). The instruction modifier selects the data interpretation mode (INT32, LO16, etc.). |
| `TTI_SFPSETCC` | Sets the condition code register based on a comparison of an SFPU local register. Mode 4 checks for negative values. |
| `TTI_SFPIADD` | Integer addition of a 12-bit signed immediate to an SFPU local register. Supports conditional execution modes (mode 1 = execute if CC not set). |
| `TTI_SFPCOMPC` | Complements (inverts) the current condition code bits. |
| `TTI_SFPMOV` | Moves/copies a value between SFPU local registers. Operates conditionally when CC is active. |
| `TTI_SFPENCC` | Disables conditional execution by clearing the condition code register. |
| `TTI_SFPSHFT` | Performs a bitwise shift on LREG (destination register). The shift amount comes from another LREG. Positive values shift left, negative values shift right. |
| `TT_SFPSTORE` | Stores a vector from an SFPU local register back to a DEST register row. |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `p_sfpu::LREG0` | Holds the value to be shifted (input A). Also holds the intermediate and final result. |
| `p_sfpu::LREG1` | Holds the shift amount (input B). Used as the shift operand for `SFPSHFT`. |
| `p_sfpu::LREG2` | Temporary: holds `shift_amount - 32` for boundary checking. |
| `p_sfpu::LCONST_0` | Hardware constant register holding zero. Used to zero out lanes with invalid shift amounts. |
| DEST registers | Tile data storage. Tiles are loaded from/stored to DEST at offsets `dst_index * 64`. |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel calls `tile_regs_acquire()` and `tile_regs_wait()` to acquire exclusive access to the DEST register file.
2. **Unpack to DEST**: `copy_tile(cb_inp0, i, i*2)` unpacks input A tile `i` from CB c_0 into DEST[i*2]. `copy_tile(cb_inp1, i, i*2+1)` unpacks input B tile `i` from CB c_1 into DEST[i*2+1]. Note: `copy_tile` is used instead of the FPU unpack path because this is an SFPU operation with `UnpackToDestFp32` mode enabled.
3. **SFPU init**: `binary_shift_tile_init()` is called (via `SHIFT_INIT` define) to configure the SFPU pipeline. This calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>()` which sets up the SFPU execution context.
4. **SFPU dispatch**: `binary_left_shift_tile<DataFormat::XXX>(i*2, i*2+1, i*2)` is called. This routes through:
   - `llk_math_eltwise_binary_sfpu_left_shift` -> `_llk_math_eltwise_binary_sfpu_params_` -> iterates over 4 faces in RC vector mode -> calls `_calculate_binary_left_shift_` per face.
5. **Per-face execution** (called 4 times by `_llk_math_eltwise_binary_sfpu_params_` in RC mode): Each call to `_calculate_binary_left_shift_` runs 8 iterations. In each iteration:
   - Load one row from DEST into LREG0 (value) and LREG1 (shift amount)
   - Check if shift amount is in valid range [0, 31]; zero out result for invalid lanes
   - Execute `SFPSHFT` to perform the left shift
   - Store result back to DEST
   - Advance the DEST row pointer
6. **Pack**: `pack_tile(i*2, cb_out0)` packs the result from DEST[i*2] into the output CB c_2.
7. **Release**: `tile_regs_commit()` and `tile_regs_release()` free the DEST register file.

#### SFPU Configuration

- **`fp32_dest_acc_en`**: Set to `true` when output data format is Float32, Int32, or UInt32. This enables 32-bit accumulation in DEST registers, which is necessary for integer operations.
- **`UnpackToDestMode::UnpackToDestFp32`**: Set for all input CBs (c_0, c_1, c_3, c_4). This configures the unpacker to send data directly to DEST in 32-bit format, bypassing the FPU. This is essential because the SFPU reads from DEST registers.
- **`INSTRUCTION_MODE`**: Determined by data type:
  - `DataFormat::UInt16` -> `InstrModLoadStore::LO16` (16-bit integer mode)
  - `DataFormat::Int32` or `DataFormat::UInt32` -> `InstrModLoadStore::INT32` (32-bit integer mode)
- **`SIGN_MAGNITUDE_FORMAT`**: Default `false`. When true, uses `INT32_2S_COMP` mode for load/store.
- **`APPROXIMATION_MODE`**: Passed through from `APPROX` compile-time flag but not actually used by the shift implementation (the shift is exact, not approximate).

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The only difference is the address modifier used in `TT_SFPLOAD`/`TT_SFPSTORE`: Wormhole uses `ADDR_MOD_3` while Blackhole uses `ADDR_MOD_7`. These control how the DEST register address auto-increments between SFPU operations. The functional behavior is identical.
- **Supported data types**: Both architectures support Int32, UInt32, and UInt16. Float types are not supported for shift operations.
- **DEST capacity**: With `fp32_dest_acc_en=true` (required for integer types), DEST can hold 2 tiles from each operand (32-bit mode), for a total of 4 tiles. With 16-bit formats, capacity doubles to 4 tiles per operand.

## Implementation Notes

1. **Boundary handling**: The SFPU kernel carefully handles out-of-range shift amounts. If `shift_amount < 0` or `shift_amount >= 32`, the result is set to 0. This is implemented using SFPU condition codes rather than branching, making it a branchless SIMD operation.

2. **Per-tile init overhead**: The `SHIFT_INIT` (`binary_shift_tile_init()`) is called inside the inner tile loop (once per tile), not once before the loop. This means the SFPU is re-initialized for every tile. This is a pattern common to the generic binary SFPU kernel to support operations that may need different init per iteration.

3. **Interleaved lock-step reads**: The reader kernel reads one tile from src0 and one from src1, waits for both, then pushes both. This ensures tiles arrive in matched pairs, which is critical for binary operations.

4. **No pre-scaling for shift**: Unlike operations like LOGADDEXP that pre-scale inputs, binary left shift does not define `SFPU_OP_INIT_PRE_IN0_0` or `SFPU_OP_INIT_PRE_IN1_0`, so the interim CBs (c_3, c_4) are not allocated and the pre-scaling code paths are compiled out.

5. **Result overwrites input A slot**: The SFPU operation writes the result to DEST[i*2] (same slot as input A). This is an in-place computation that avoids needing additional DEST register space for the output.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary elementwise SFPU program factory work? What kernels does it use (reader, compute, writer)? How does it handle different binary operations like left_shift? What is the core distribution strategy?"
   **Reason**: Needed to understand the overall architecture of the SFPU binary program factory before diving into code.
   **Key Findings**: Confirmed three-kernel architecture (reader/compute/writer), identified that LEFT_SHIFT maps to `SfpuBinaryOp::LEFT_SHIFT` through `OpConfig`, uses `binary_shift_tile_init()` and `binary_left_shift_tile` functions, and distributes work across `all_device_cores`.

2. **Query**: "What is the binary left shift SFPU operation in TTNN? How is it implemented at the kernel level? What SFPU instructions does it use?"
   **Reason**: Needed specific details about the left shift SFPU implementation, instruction chain, and data format constraints.
   **Key Findings**: Identified the call chain from `binary_left_shift_tile` -> `llk_math_eltwise_binary_sfpu_left_shift` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_binary_left_shift` -> `_calculate_binary_left_shift_`. Confirmed supported data formats (Int32, UInt32, UInt16) and instruction mode selection logic.

3. **Query**: "Where is the _calculate_binary_left_shift_ function defined? What does it do at the SFPU instruction level?"
   **Reason**: Needed the actual SFPU microcode implementation details since DeepWiki references were to the wrapper, not the core implementation.
   **Key Findings**: Function is in `ckernel_sfpu_shift.h` in the tt-llk submodule. Identified the full instruction sequence: SFPLOAD -> boundary check via SFPSETCC/SFPIADD/SFPCOMPC/SFPMOV/SFPENCC -> SFPSHFT -> SFPSTORE. Confirmed Blackhole uses ADDR_MOD_7 and Wormhole uses ADDR_MOD_3.

### Confluence References

None consulted. DeepWiki and direct source code reading provided sufficient detail for the SFPU instructions used in this operation.

### Glean References

None consulted. The SFPU shift instructions are well-documented in the open-source tt-llk codebase.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 307-314)
   **Reason**: Needed to understand how the `SHIFT_INIT` and `BINARY_SFPU_OP` defines are generated for LEFT_SHIFT.
   **Key Information**: `SHIFT_INIT` = `binary_shift_tile_init();`, `BINARY_SFPU_OP` = `binary_left_shift_tile<DataFormat::XXX>(i*2, i*2+1, i*2);` where XXX depends on input dtype. The `idst1="i*2"` and `idst2="i*2+1"` pattern places inputs at interleaved DEST positions.

2. **Source**: `tt_metal/hw/inc/api/compute/binary_shift.h`
   **Reason**: Needed the public API documentation for `binary_left_shift_tile`.
   **Key Information**: Confirmed the function signature `binary_left_shift_tile<DataFormat>(idst0, idst1, odst)`, supported formats (Int32, UInt32, UInt16), and DST register buffer constraints (max 2 tiles per operand in 32-bit mode, 4 in 16-bit mode).

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Needed to understand how `_llk_math_eltwise_binary_sfpu_params_` dispatches the SFPU function across tile faces.
   **Key Information**: In RC vector mode (default), the function iterates over 4 faces, calling the SFPU function once per face, with `TTI_SETRWC` advancing the DEST row pointer by 8 between faces. This means `_calculate_binary_left_shift_` with ITERATIONS=8 processes 8 rows per call, and 4 calls cover 32 rows = one full 32x32 tile.
