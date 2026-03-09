# Binary Left Shift Implementation Analysis

## Overview

The binary left shift operation performs element-wise bitwise left shift: `y = x0 << x1`, where `x0` is the value to shift and `x1` is the shift amount. It operates on integer data types (Int32, UInt32, UInt16) using the SFPU vector unit on Tenstorrent hardware. The operation includes bounds checking -- if the shift amount is negative or >= 32, the result is clamped to zero.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `physical_volume / TILE_HW` (total tiles across all tensor elements) |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_size` tiles within each block |

## Tensor Format and Layout

### Input Tensors

| Property | Input Tensor A (value) | Input Tensor B (shift amount) |
|----------|------------------------|-------------------------------|
| **Logical shape** | Arbitrary (must match B or be broadcastable via caller) | Arbitrary (must match A) |
| **Dimension convention** | NHWC | NHWC |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 | DRAM or L1 |
| **Data type** | Int32, UInt32, or UInt16 | Same as Input A |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as Input A |
| **Dimension convention** | NHWC |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED or SHARDED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as Input A |

### Layout Transformations

No explicit tilize/untilize or format conversions are performed within the program factory. Data must arrive in TILE_LAYOUT. The unpack-to-dest mode is set to `UnpackToDestFp32` for all CBs (since LEFT_SHIFT is not POWER), which ensures 32-bit precision in the DEST register file.

## Data Flow Pattern

1. **Reader kernel** reads one tile at a time from each input tensor (A into CB c_0, B into CB c_1) via NoC. If inputs are sharded, the reader simply marks all shard tiles as available in the respective CBs.
2. **Compute kernel** waits for tiles in CB c_0 and CB c_1, copies them into DEST registers (input A at even slots `i*2`, input B at odd slots `i*2+1`), calls the `SHIFT_INIT` (one-time SFPU pipeline config), then dispatches `binary_left_shift_tile<DataFormat::X>(i*2, i*2+1, i*2)` which runs the SFPU left shift microcode. The result overwrites the input A slot in DEST. The result is packed to CB c_2.
3. **Writer kernel** reads one tile at a time from CB c_2 and writes to the output buffer via NoC. If the output is sharded, the writer simply waits for all tiles to appear in CB c_2 (which is aliased to the output shard buffer).

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input A staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_1 | cb_src1 | Input B staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded) | 1 tile | Double (interleaved) / Single (sharded) | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles (interleaved) or `num_tiles_per_shard` (sharded/block-width) | 1 tile | Double (interleaved) / Single (sharded) | Compute | Writer | Program |

Notes:
- For left shift, `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` are NOT defined, so intermediate CBs c_3 and c_4 are NOT created.
- Capacity formula for interleaved: `2 * max_block_size` tiles where `max_block_size` = 1 for interleaved (non-sharded) path.
- When sharded, CB c_0/c_1 are globally allocated to the input tensor shard buffers. CB c_2 is globally allocated to the output shard buffer when output is sharded.

## Pipeline Pattern Summary

- **Interleaved path**: All three CBs have capacity = 2 tiles with block size = 1 tile, enabling **double-buffered** overlap between reader and compute, and between compute and writer.
- **Sharded path**: CBs are sized to hold all shard tiles at once. Since the reader does no work (data is already in L1), there is no reader/compute overlap. The compute and writer similarly have no overlap for sharded output.

## Index Calculations

The reader kernel uses `TensorAccessor` for DRAM-interleaved access, which maps a linear tile ID to the appropriate DRAM bank and offset. The tile ID is computed as a simple linear index starting from `start_id`, incremented by 1 for each tile read. There is no complex index remapping for the non-sharded path.

For block/width-sharded tensors, the reader iterates over `block_height` rows and `block_width` columns, computing tile IDs as:
```
tile_id = row_start_tile_id + w
row_start_tile_id += num_cores_y * block_width  (per row)
```

The runtime args helper computes `start_id` per core as `num_tiles_read` (cumulative count) for interleaved, or a shard-grid-based formula for sharded:
```
start_id = (core_idx / num_shards_per_width) * (block_height * block_width * num_shards_per_width)
         + (core_idx % num_shards_per_width) * block_width
```

## Memory Access Patterns

### Read Pattern
- **Interleaved**: Sequential tile reads from DRAM. Each tile is read via `noc_async_read_tile` with a barrier after each tile pair (one from A, one from B).
- **Sharded**: No reads -- data is already in L1 shard buffers. The reader kernel just calls `cb_reserve_back` / `cb_push_back` to make tiles visible.

### Write Pattern
- **Interleaved**: Sequential tile writes to DRAM. Each tile is written via `noc_async_write_page` with a flush after each tile, followed by a final barrier.
- **Sharded**: No writes -- output CB is aliased to the output shard buffer. The writer kernel just calls `cb_wait_front`.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | Depends on `operation_attributes.worker_grid`; typically 2D |
| **Grid dimensions** | Determined by the worker grid or shard spec grid |
| **Total cores** | `num_cores_total` = grid_x * grid_y |
| **Work per core** | `num_tiles_per_core_group_1` (most cores) or `num_tiles_per_core_group_2` (remainder cores) |
| **Load balancing** | Two-group split via `split_work_to_cores`: group 1 gets `ceil(total_tiles / num_cores)` tiles, group 2 gets `floor(total_tiles / num_cores)` tiles |

For sharded tensors, all cores in the shard grid get an equal number of tiles (`shard_shape[0] * shard_shape[1] / TILE_HW`), so there is no remainder group.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | block_or_width_sharded | uint32_t | 1 if input uses block or width sharding, 0 otherwise |
| 1+ | TensorAccessorArgs(src0) | varies | Accessor parameters for input A buffer (omitted if IN0_SHARDED) |
| N+ | TensorAccessorArgs(src1) | varies | Accessor parameters for input B buffer (omitted if IN1_SHARDED) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (c_2) |
| 1+ | TensorAccessorArgs(dst) | varies | Accessor parameters for output buffer |

#### Compute Kernel

The compute kernel has no indexed compile-time arguments. Instead, it receives preprocessor defines:
- `SHIFT_INIT` = `binary_shift_tile_init();` -- SFPU pipeline initialization for shift ops.
- `BINARY_SFPU_OP` = `binary_left_shift_tile<DataFormat::X>(i*2, i*2+1, i*2);` -- the actual SFPU dispatch call, where `X` is `Int32`, `UInt32`, or `UInt16` depending on input dtypes.
- `fp32_dest_acc_en` = true (set when output dtype is Int32/UInt32).
- `UnpackToDestMode::UnpackToDestFp32` for all CB indices.

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src0_addr | uint32_t | Source buffer address for input A |
| 1 | src1_addr | uint32_t | Source buffer address for input B |
| 2 | num_tiles | uint32_t | Number of tiles this core processes |
| 3 | start_id | uint32_t | Starting tile ID for this core |
| 4 | block_height | uint32_t | Shard block height in tiles (0 for interleaved) |
| 5 | block_width | uint32_t | Shard block width in tiles (0 for interleaved) |
| 6 | num_cores_y | uint32_t | Number of shard columns for strided access |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process |
| 1 | per_core_block_size | uint32_t | Number of tiles per block |

#### Writer Kernel (interleaved output)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | num_pages | uint32_t | Number of tiles to write |
| 2 | start_id | uint32_t | Starting tile ID for output |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_binary_interleaved_start_id | RISCV_0 | NOC0 | DRAM/L1 (src0, src1) | CB c_0, CB c_1 | Read tiles from both inputs |
| eltwise_binary_sfpu_kernel | RISCV_2 (compute) | N/A | CB c_0, CB c_1 | CB c_2 | Copy to DEST, SFPU left shift, pack to output CB |
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 (dst) | Write output tiles |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`
- **Key Logic**: Supports both interleaved and sharded modes via compile-time defines (`IN0_SHARDED`, `IN1_SHARDED`). For interleaved, reads tiles one at a time using `noc_async_read_tile` with `TensorAccessor`. For sharded, simply marks all tiles as available via `cb_reserve_back`/`cb_push_back`. Supports block/width-sharded 2D iteration pattern.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Standard single-tile write loop. For sharded output, just waits for tiles. For interleaved, writes one tile at a time with `noc_async_write_page` using `TensorAccessor`.

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
#include "api/compute/binary_shift.h"      // provides binary_left_shift_tile, binary_shift_tile_init
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

// PRE_SCALE is true if either SFPU_OP_INIT_PRE_IN0_0 or SFPU_OP_INIT_PRE_IN1_0 is defined.
// For LEFT_SHIFT, neither is defined, so PRE_SCALE is false.
#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);   // number of tile blocks to process
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);  // tiles per block

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // CB for input A (value)
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // CB for input B (shift amount)

    // For LEFT_SHIFT, SFPU_OP_INIT_PRE_IN0_0 is not defined, so cb_inp0 == cb_in0
#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;
#else
    constexpr auto cb_inp0 = cb_in0;
#endif

    // For LEFT_SHIFT, SFPU_OP_INIT_PRE_IN1_0 is not defined, so cb_inp1 == cb_in1
#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;
#else
    constexpr auto cb_inp1 = cb_in1;
#endif

    constexpr auto cb_out0 = tt::CBIndex::c_2;  // CB for output

    unary_op_init_common(cb_in0, cb_out0);  // initialize unpack/pack pipeline for the CB pair

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));  // not active for LEFT_SHIFT
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // PRE_SCALE sections are skipped for LEFT_SHIFT (no prescaling needed)

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);
#endif

        // [SFPU_OP_INIT_PRE_IN0_0 block omitted -- not active for LEFT_SHIFT]
        // [SFPU_OP_INIT_PRE_IN1_0 block omitted -- not active for LEFT_SHIFT]

        // Wait for input tiles from both CBs
        cb_wait_front(cb_inp0, per_core_block_size);  // wait for input A tiles
        cb_wait_front(cb_inp1, per_core_block_size);  // wait for input B tiles
        cb_reserve_back(cb_out0, per_core_block_size); // reserve output space

        tile_regs_acquire();  // acquire DEST register file for writing
        tile_regs_wait();     // wait until DEST registers are available

        // Copy input A tiles into DEST at even indices (0, 2, 4, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);  // configure unpack for cb_inp0's data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);  // copy tile i from cb_inp0 to DEST[i*2]
        }

        // Copy input B tiles into DEST at odd indices (1, 3, 5, ...)
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);  // reconfigure unpack for cb_inp1's data type
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);  // copy tile i from cb_inp1 to DEST[i*2+1]

            // For LEFT_SHIFT, SHIFT_INIT is defined as binary_shift_tile_init()
            // This is called once per tile to configure the SFPU shift pipeline
#ifdef SHIFT_INIT
            SHIFT_INIT    // expands to: binary_shift_tile_init();
#endif

            // BINARY_SFPU_OP is the main computation dispatch
            // Expands to: binary_left_shift_tile<DataFormat::Int32>(i*2, i*2+1, i*2);
            // This calls the SFPU kernel to compute DEST[i*2] = DEST[i*2] << DEST[i*2+1]
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP
#endif

            // Pack the result from DEST[i*2] into the output CB
            pack_tile(i * 2, cb_out0);
        }
        tile_regs_commit();   // signal that DEST writes are complete
        tile_regs_release();  // release DEST register file

        cb_pop_front(cb_inp0, per_core_block_size);   // free consumed input A tiles
        cb_pop_front(cb_inp1, per_core_block_size);   // free consumed input B tiles
        cb_push_back(cb_out0, per_core_block_size);   // publish output tiles for writer
    }
}
```

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
- **Blackhole**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h`
- **Wormhole B0**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_shift.h`

The Blackhole version is shown below (Wormhole is identical except it uses `ADDR_MOD_3` instead of `ADDR_MOD_7` for load/store address modifiers).

#### Annotated SFPU Kernel Source

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _calculate_binary_left_shift_(
    const std::uint32_t dst_index_in0,    // DEST tile index for input A (value to shift)
    const std::uint32_t dst_index_in1,    // DEST tile index for input B (shift amount)
    const std::uint32_t dst_index_out)    // DEST tile index for output (result overwrites in0)
{
    // Validate instruction mode at compile time
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE),
        "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Determine load/store format: use 2's complement if SIGN_MAGNITUDE_FORMAT, otherwise use the
    // native instruction mode (INT32 for Int32/UInt32, LO16 for UInt16)
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    // SFPU processes a tile face (32 elements) per iteration.
    // A 32x32 tile has 4 faces, so the _llk_math_eltwise_binary_sfpu_params_ wrapper calls this
    // function 4 times (once per face), advancing dst_reg by 16 rows each time.
    // Within each call, ITERATIONS=8 processes 8 groups of 4 rows = 32 rows = 2 faces worth,
    // but the params wrapper handles the face iteration with SETRWC.
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Each tile occupies 64 rows in DEST (4 faces x 16 rows per face in SFPU addressing)
        constexpr std::uint32_t dst_tile_size = 64;

        // Load 32 elements of input A (value) from DEST into SFPU local register LREG0
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);

        // Load 32 elements of input B (shift amount) from DEST into SFPU local register LREG1
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);

        // --- Bounds checking: if shift_amount < 0 OR shift_amount >= 32, result = 0 ---

        // Set lane flags where LREG1 (shift amount) < 0 (check sign bit)
        // Mode 4 = set flags based on sign of VC (LREG1)
        TTI_SFPSETCC(0, p_sfpu::LREG1, p_sfpu::LREG0, 4);

        // LREG2 = LREG1 + (-32) = shift_amount - 32
        // Mode 1: integer add with immediate, result in LREG2
        // 0xFE0 is the 11-bit signed representation of -32
        TTI_SFPIADD(0xFE0, p_sfpu::LREG1, p_sfpu::LREG2, 1);

        // Complement condition codes: flip the lane flags
        // After SFPSETCC set flags for shift<0, SFPCOMPC flips to get shift>=0 lanes
        // Combined with the next SFPIADD result, this implements: flag = (shift<0) || (shift>=32)
        TTI_SFPCOMPC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // For flagged lanes (where shift is out of bounds), set LREG0 = 0 (LCONST_0)
        // This zeroes the result for invalid shift amounts
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // Disable conditional execution (clear lane flag usage)
        TTI_SFPENCC(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);

        // --- Perform the actual left shift ---

        // LREG0 = LREG0 << LREG1  (shift value left by shift amount)
        // SFPSHFT shifts LREG0 (dest) by the amount in LREG1 (src); positive = left shift
        TTI_SFPSHFT(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);

        // Store the result from LREG0 back to DEST at the output tile position
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);

        // Advance the DEST register row pointer for the next iteration
        sfpi::dst_reg++;
    }
}
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOAD` | Loads 32 elements from a DEST register row into an SFPU local register (LREG). Supports integer format modes (INT32, LO16). Uses address modifier for row advancement. |
| `TT_SFPSTORE` | Stores 32 elements from an SFPU local register back to a DEST register row. Same format modes as SFPLOAD. |
| `TTI_SFPSETCC` | Sets per-lane condition flags based on a comparison. Mode 4 checks the sign bit of the source register (flags lanes where value < 0). |
| `TTI_SFPIADD` | Integer add with an 11-bit signed immediate. `0xFE0` = -32. Mode 1 stores result in a destination register. Used to compute `shift_amount - 32` for bounds checking. |
| `TTI_SFPCOMPC` | Complements (inverts) the per-lane condition code flags. Used to implement the "else" branch of the bounds check. |
| `TTI_SFPMOV` | Moves/copies data between SFPU local registers. Here used to zero out LREG0 from LCONST_0 for lanes with out-of-bounds shift amounts. |
| `TTI_SFPENCC` | Enables/disables per-lane conditional execution. Mode 0 with immediate 0 disables conditional execution (all lanes active again). |
| `TTI_SFPSHFT` | Bitwise shift instruction. Shifts the destination register (LREG0) by the amount in the source register (LREG1). Positive values = left shift, negative values = right shift. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | Input A value (loaded from DEST), then overwritten with result. Also used as output. |
| `LREG1` | Input B shift amount (loaded from DEST). Used as shift operand for SFPSHFT. |
| `LREG2` | Scratch register for bounds check result (`shift_amount - 32`). |
| `LCONST_0` | Constant zero register, used to zero out results for invalid shift amounts. |
| `dst_reg` | SFPU's internal DEST row pointer, incremented each iteration to advance through the tile face. |

#### SFPU Execution Flow

1. **Tile acquisition**: The compute kernel copies tiles from CBs c_0 and c_1 into DEST registers using `copy_tile`. Input A is placed at DEST index `i*2`, input B at DEST index `i*2+1`.

2. **SFPU dispatch**: `binary_left_shift_tile<DataFormat::X>(i*2, i*2+1, i*2)` is called, which routes through:
   - `binary_left_shift_tile` (in `binary_shift.h`) calls `llk_math_eltwise_binary_sfpu_left_shift`
   - `llk_math_eltwise_binary_sfpu_left_shift` (in `llk_math_eltwise_binary_sfpu_shift.h`) calls `_llk_math_eltwise_binary_sfpu_params_` with `calculate_binary_left_shift` as the callable
   - `_llk_math_eltwise_binary_sfpu_params_` (in `llk_math_eltwise_binary_sfpu_params.h`) iterates over 4 tile faces in `VectorMode::RC` mode, calling `_calculate_binary_left_shift_` for each face and advancing DEST row pointers with `TTI_SETRWC`

3. **Per-face SFPU execution** (the `_calculate_binary_left_shift_` function runs 8 iterations per face):
   - Load 32 elements of value (LREG0) and shift amount (LREG1) from DEST
   - Bounds check: set flags for `shift < 0`, compute `shift - 32`, complement flags to get `shift < 0 || shift >= 32`
   - Zero out result for flagged (out-of-bounds) lanes
   - Disable conditional execution
   - Perform left shift: `LREG0 = LREG0 << LREG1`
   - Store result back to DEST at the output position
   - Increment `dst_reg` to advance to next row

4. **Pack**: After SFPU completes, `pack_tile(i*2, cb_out0)` packs the result from DEST back into the output circular buffer.

#### SFPU Configuration

- **Initialization**: `binary_shift_tile_init()` calls `llk_math_eltwise_binary_sfpu_shift_init()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>()`. This sets up generic binary SFPU infrastructure without any operation-specific LUT configuration.
- **Approximation mode**: The `APPROX` template parameter is passed through but not used meaningfully for shift operations (no LUT-based approximation needed for integer bit manipulation).
- **Data format**: Determined at compile time based on input dtypes. Maps `UInt16` to `InstrModLoadStore::LO16` (16-bit integer load/store mode) and `Int32`/`UInt32` to `InstrModLoadStore::INT32` (32-bit integer mode).
- **fp32_dest_acc_en**: Set to `true` when output is Int32 or UInt32, ensuring DEST operates in 32-bit accumulation mode.
- **UnpackToDestFp32**: All CBs are configured for FP32 unpack-to-dest mode (since the operation is not POWER type), which ensures full 32-bit precision when moving data from CBs to DEST.

#### Hardware Compatibility Notes

The left shift SFPU kernel is functionally identical between Wormhole B0 and Blackhole architectures. The only difference is the **address modifier** used in `TT_SFPLOAD` and `TT_SFPSTORE`:
- **Wormhole B0**: Uses `ADDR_MOD_3`
- **Blackhole**: Uses `ADDR_MOD_7`

This reflects different DEST register addressing configurations between the two architectures but does not affect the computation logic.

Both architectures include `_calculate_logical_right_shift_` in addition to the left and right shift functions, but the Wormhole B0 version does not expose `_calculate_logical_right_shift_` through the `ckernel_sfpu_shift.h` wrapper (it is directly referenced from the LLK layer).

## Implementation Notes

1. **No prescaling**: Unlike operations like LOGADDEXP or LDEXP, binary left shift does not require any input prescaling. The `SFPU_OP_INIT_PRE_IN0_0` and `SFPU_OP_INIT_PRE_IN1_0` defines are not set, so intermediate CBs c_3 and c_4 are not allocated, and the prescaling code blocks in the compute kernel are compiled out.

2. **Bounds checking in SFPU**: The kernel defensively handles out-of-range shift amounts. If the shift amount is negative or >= 32, the result is set to zero. This is implemented using SFPU conditional execution (lane flags) rather than branching, maintaining SIMD efficiency.

3. **DEST register interleaving**: Input A and B tiles are placed at alternating DEST indices (`i*2` and `i*2+1`). The result overwrites the input A position (`i*2`), so only input A's DEST slot is packed to the output CB. This interleaving pattern is a standard approach for binary SFPU operations that need both operands in DEST simultaneously.

4. **Block size = 1 for interleaved**: When tensors are interleaved (not sharded), `max_block_size` is always 1, meaning the compute kernel processes one tile per block. The "block count" equals the number of tiles assigned to the core.

5. **Sharding support**: The program factory transparently handles height-sharded, width-sharded, and block-sharded memory layouts. For sharded inputs, CBs are globally allocated to the shard buffer addresses, eliminating DRAM-to-L1 data movement entirely.

6. **SHIFT_INIT called per tile**: The `binary_shift_tile_init()` is called inside the inner loop (once per tile), not outside. This is because the `copy_tile_to_dst_init_short_with_dt` calls above may reconfigure the unpack pipeline, and the shift init restores SFPU-specific configuration.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the binary eltwise SFPU program factory work in TTNN? What kernels does it use?"
   **Reason**: Needed to understand the overall structure of the binary SFPU program factory, kernel selection, and broadcasting support.
   **Key Findings**: Identified the three kernels (reader, compute, writer), the `ElementWiseMultiCoreSfpu` factory class, and how defines are used to specialize the generic compute kernel.

2. **Query**: "How is the binary left shift operation implemented as an SFPU operation?"
   **Reason**: Needed to trace the call chain from TTNN Python API down to the SFPU kernel.
   **Key Findings**: Identified the call chain: `binary_left_shift_tile` -> `llk_math_eltwise_binary_sfpu_left_shift` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_binary_left_shift` -> `_calculate_binary_left_shift_`. Confirmed Int32/UInt32/UInt16 support.

3. **Query**: "What do SFPLOAD, SFPSTORE, SFPSETCC, SFPIADD, SFPCOMPC, SFPMOV, SFPENCC, SFPSHFT do?"
   **Reason**: Needed to annotate each SFPU instruction in the kernel with accurate descriptions.
   **Key Findings**: SFPLOAD/SFPSTORE move data between DEST and SFPU local registers. SFPSETCC sets per-lane flags for conditional execution. SFPIADD does integer add with immediate. SFPCOMPC complements condition codes. SFPMOV copies registers. SFPENCC enables/disables conditional execution. SFPSHFT performs bitwise shift.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (lines 307-314)
   **Reason**: Needed to understand exactly which preprocessor defines are set for LEFT_SHIFT.
   **Key Information**: LEFT_SHIFT sets `SHIFT_INIT` = `binary_shift_tile_init();` and `BINARY_SFPU_OP` = `binary_left_shift_tile<DataFormat::X>(i*2, i*2+1, i*2);`. Data format is selected based on input dtypes.

2. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h`
   **Reason**: Needed to understand how the params wrapper iterates over tile faces.
   **Key Information**: In `VectorMode::RC` (default), the wrapper calls the SFPU function 4 times (once per face), advancing DEST row pointers by 16 rows between faces using `TTI_SETRWC`.

3. **Source**: `tt_metal/hw/inc/api/compute/binary_shift.h`
   **Reason**: Needed to see the API-level `binary_left_shift_tile` function signature and documentation.
   **Key Information**: Takes `idst0` (first operand), `idst1` (second operand), `odst` (output) as DEST indices. Templated on `DataFormat`. Supported formats: Int32, UInt32, UInt16.
