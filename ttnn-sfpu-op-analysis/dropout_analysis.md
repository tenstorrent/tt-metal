# Dropout Implementation Analysis

## Overview
The dropout operation performs element-wise stochastic regularization on a tensor: each element is either zeroed out with probability `prob` or scaled by a factor `scale = 1 / (1 - prob)` (inverted dropout). The operation uses a hardware PRNG on each SFPU lane to generate random numbers, compares them against the dropout probability, and conditionally zeros elements.

**Program factory path**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp`

The operation supports two program factory variants:
- `DropoutProgramFactory` -- single-device execution
- `DropoutMeshWorkloadFactory` -- multi-device mesh execution with per-device seed offsets (seed += device_id)

This analysis covers `DropoutProgramFactory`.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `input.physical_volume() / TILE_HW` |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` (always 1) tiles per block |

Each tile is independently processed: loaded from DEST, scaled, subjected to random dropout masking, and stored back. The block dimension is hardcoded to 1, so the outer loop simply iterates over all tiles assigned to the core.

## Tensor Format and Layout

| Property | Input Tensor | Output Tensor |
|----------|--------------|---------------|
| **Logical shape** | Arbitrary (flattened to tiles) | Same as input |
| **Dimension convention** | N/A (treated as flat tile stream) | N/A |
| **Tensor layout** | TILE_LAYOUT | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED | INTERLEAVED |
| **Buffer type** | DRAM (via TensorAccessor) | DRAM (via TensorAccessor) |
| **Data type** | Determined by `input.dtype()` | Determined by `output_dtype` param |

### Layout Transformations
No tilize/untilize or format conversions are performed. Input and output data formats may differ (e.g., BFLOAT16 input, BFLOAT16 output) but no explicit conversion logic is present in the program factory -- the hardware handles format conversion during SFPLOAD/SFPSTORE via the `data_format` configuration on circular buffers.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer via TensorAccessor) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_tile`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, `dropout_tile`, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_reserve_back(c_2, 1)`, `cb_push_back(c_2, 1)` |
| 3 | Writer | CB c_2 | DRAM (dst_buffer via TensorAccessor) | `cb_wait_front(c_2, 1)`, `noc_async_write_tile`, `noc_async_write_barrier`, `cb_pop_front(c_2, 1)` |

Data flows tile-by-tile through the pipeline: Reader fetches one tile from DRAM into CB c_0, Compute copies it to DEST registers, applies dropout (scale + random mask), packs to CB c_2, and Writer drains CB c_2 to DRAM.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

## Pipeline Pattern Summary
Both circular buffers are double-buffered (capacity = 2 tiles, block size = 1 tile). This allows the reader to pre-fetch the next tile while compute processes the current one, and compute to produce the next output tile while the writer drains the current one. The pipeline supports overlap between all three kernel stages.

## Index Calculations
The program factory uses `TensorAccessor` for address computation. The reader and writer kernels receive a `start_id` (tile offset) and `num_tiles` count. Tile indices are sequential: `start_id` to `start_id + num_tiles - 1`. The `TensorAccessor` maps each linear tile index to the correct DRAM bank address via `noc_async_read_tile` / `noc_async_write_tile`.

No complex index transformations are needed because the tensor is treated as a flat stream of tiles with interleaved memory layout.

## Memory Access Patterns

### Read Pattern
Sequential tile reads from DRAM. Each tile is read individually via `noc_async_read_tile(i, s, l1_write_addr)` with a barrier after each read. Tiles are accessed in ascending linear index order (or descending if `BACKWARDS` is defined, though the program factory does not define this).

### Write Pattern
Sequential tile writes to DRAM. Each tile is written individually via `noc_async_write_tile(i, s, l1_read_addr)` with a barrier after each write. Same ascending order as reader.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major traversal) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | min(num_tiles, grid_size.x * grid_size.y) |
| **Work per core** | `ceil(num_tiles / num_cores)` for group 1, `floor(num_tiles / num_cores)` for group 2 |
| **Load balancing** | Two-group split via `split_work_to_cores` |

Core iteration is column-major: `core = {i / num_cores_y, i % num_cores_y}`. Group 1 cores each handle `num_tiles_per_core_group_1` tiles (one more than group 2). Group 2 cores handle `num_tiles_per_core_group_2` tiles. If tiles divide evenly, group 2 is empty.

Separate compute kernels are created for group 1 and group 2 because `per_core_block_cnt` (the tile count) is a compile-time argument.

## Arguments

### Compile-Time Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in0 | uint32_t | Circular buffer index for input (c_0) |
| 1+ | TensorAccessorArgs | uint32_t[] | Source buffer accessor parameters (appended by `TensorAccessorArgs`) |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Circular buffer index for output (c_2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Destination buffer accessor parameters |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process (differs between group 1 and group 2) |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1) |
| 2 | prob_int | uint32_t | Dropout probability as integer: `(double)INT_MAX * prob` |
| 3 | uscale | uint32_t | Scale factor as bit-cast uint32_t of float32 |

### Runtime Arguments

#### Reader Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_tiles | uint32_t | Number of tiles this core processes |
| 2 | start_id | uint32_t | Starting tile index offset for this core |

#### Writer Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_tiles | uint32_t | Number of tiles this core processes |
| 2 | start_id | uint32_t | Starting tile index offset for this core |

#### Compute Kernel
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | seed | uint32_t | PRNG seed for dropout random number generation |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | BRISC (RISCV_0) | NOC0 | DRAM (src) | CB c_0 | Read tiles via TensorAccessor |
| compute | MATH (RISCV_2) | N/A | CB c_0 | CB c_2 | copy_tile, dropout_tile (SFPU), pack_tile |
| writer | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM (dst) | Write tiles via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp`
- **Key Logic**: Standard single-tile interleaved reader. Uses `TensorAccessor` for DRAM addressing. Reads tiles sequentially from `start_id` to `start_id + num_tiles - 1`. Supports optional `BACKWARDS` mode (not used by this program factory). Each tile is read with a NoC barrier before pushing to CB.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp`
- **Key Logic**: Standard single-tile interleaved writer. Mirrors reader structure. Supports optional `OUT_SHARDED` mode (not used by this program factory) where it simply waits for all tiles in CB without writing to DRAM. In the interleaved path, drains tiles one at a time with NoC write barriers.

### Compute Kernel
This section combines the full annotated source code of the compute kernel with architectural analysis.

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`

#### Annotated Compute Kernel Source
```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/dropout.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);   // total number of tile-blocks this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);   // tiles per block; always 1 for this operation
    uint32_t int_probability = get_compile_time_arg_val(2);      // dropout probability as integer (prob * INT_MAX)
    uint32_t int_scale_factor = get_compile_time_arg_val(3);     // scale factor as bit-cast float32 -> uint32_t

    uint32_t seed = get_arg_val<uint32_t>(0);                    // PRNG seed, passed as runtime arg so it can change per invocation

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);              // configures unpack (from c_0) and pack (to c_2) pipelines for SFPU mode
    dropout_kernel_init(seed);                                    // seeds the hardware PRNG via init_prng_seed; includes 600 NOP wait for seed propagation
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve space in output CB for one block (1 tile)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();                                  // acquire exclusive access to DEST registers for math

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);                 // block until reader has produced 1 tile in input CB

            copy_tile(tt::CBIndex::c_0, 0, 0);                  // unpack tile 0 from c_0 into DEST register 0

            dropout_tile(0, int_probability, int_scale_factor);  // apply dropout SFPU operation on DEST[0]: scale then conditionally zero

            tile_regs_commit();                                   // signal that math is done, DEST is ready for packer

            tile_regs_wait();                                     // wait for packer to be ready to consume DEST

            pack_tile(0, tt::CBIndex::c_2);                      // pack DEST[0] into output CB c_2

            cb_pop_front(tt::CBIndex::c_0, 1);                  // free the consumed input tile from c_0

            tile_regs_release();                                  // release DEST registers for next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);     // publish the produced block (1 tile) to writer
    }
}
```

### SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File
`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h`
(Identical implementation for Blackhole: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h`)

#### Annotated SFPU Kernel Source
```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, std::uint32_t probability, std::uint32_t scale)
{
    // SFPU microcode

    // Load the 32-bit scale factor into LREG1 in two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);       // load low 16 bits of scale into LREG1 (mode 10 = raw low half)
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);            // load high 16 bits of scale into LREG1 (mode 8 = raw high half)

    // Load the 32-bit probability threshold into LREG2 in two 16-bit halves
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);  // load low 16 bits of probability into LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);      // load high 16 bits of probability into LREG2

#pragma GCC unroll 0                                         // disable loop unrolling; process 8 iterations (one per tile face row-pair)
    for (int d = 0; d < iterations; d++)
    {
        ////////////////////////
        // Scale samples
        // dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);
        ///////////////////////
        TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);              // load 32 elements from current DEST row-pair into LREG0 (mode 0 = no conversion, addr_mod 3 = auto-increment)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                                                             // LREG0 = LREG0 * LREG1 (element * scale_factor); result in LREG0

        ////////////////////////
        // Instruction SFPMOV generates a uint32_t pseudorandom number
        // when instr_mod1 = 8 and lreg_c = 9.
        // Arguments: (imm12_math, lreg_c, lreg_dest, instr_mod1)
        // Unset sign-bit for easy comparison with probability
        ////////////////////////
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);               // generate PRNG value into LREG3 (lreg_c=9 triggers PRNG advance, instr_mod1=8 enables PRNG mode)
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);// clear sign bit of LREG3 (make non-negative for unsigned comparison with probability)

        ////////////////////////
        // Drop samples
        // v_if (rand < probability)
        //   dst_reg[0] = vConst0;
        ///////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10); // integer subtract: LREG2 - LREG3 (probability - rand); sets per-lane condition flags
                                                             // instr_mod1=10: sets lane flags based on result sign (flag set if probability > rand, i.e., element should be KEPT)
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);// conditionally move 0.0 into LREG0 for lanes where flag is NOT set (rand >= probability => drop)
        TTI_SFPENCC(0, 0, 0, 0);                            // disable conditional execution (clear lane flag predication)
        TTI_SFPSTORE(0, 0, 3, 0);                           // store LREG0 back to DEST (mode 0, addr_mod 3 = auto-increment); contains either scaled value or 0.0

        sfpi::dst_reg++;                                     // advance DEST register pointer to next row-pair (32 elements)
    }
}

inline void _init_dropout_(const std::uint32_t seed)
{
    init_prng_seed(seed);                                    // write seed to PRNG_SEED config register; waits 600 NOPs for propagation
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOADI` | Loads a 16-bit immediate value into a specified half (high or low) of an LREG. Mode 10 = low 16 bits, mode 8 = high 16 bits. Used to construct 32-bit scale and probability values. |
| `SFPLOAD` | Loads 32 datums from the current DEST register row-pair into an LREG. Mode 0 = no type conversion. addr_mod 3 = auto-increment addressing. |
| `SFPMUL` | Floating-point multiply: `VD = VA * VB`. Multiplies each element by the scale factor. |
| `SFPMOV` (PRNG mode) | With `lreg_c=9` and `instr_mod1=8`, advances the per-lane PRNG and writes the pseudorandom uint32 into the destination LREG. |
| `SFPSETSGN` | Sets/clears the sign bit. With `instr_mod1=1`, clears the sign bit (absolute value), making the random number non-negative for comparison. |
| `SFPIADD` | Integer add/subtract with flag setting. With `instr_mod1=10`, computes `LREG2 - LREG3` (probability - random) and sets per-lane condition flags based on the result sign. |
| `SFPMOV` (conditional) | When lane flags are active, conditionally moves `LCONST_0` (0.0) into LREG0 for lanes where the condition is false (random >= probability, meaning the element is dropped). |
| `SFPENCC` | Disables vector conditional execution by clearing the `UseLaneFlagsForLaneEnable` state. |
| `SFPSTORE` | Stores 32 datums from an LREG back to the DEST register row-pair. Mode 0 = no type conversion. addr_mod 3 = auto-increment. |
| `SFPNOP` | No-operation. Used 600 times in `init_prng_seed` to wait for seed propagation to the PRNG hardware. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register: holds the current 32 elements loaded from DEST, then the scaled (or zeroed) result |
| **LREG1** | Holds the 32-bit float scale factor (loaded once before the loop, reused every iteration) |
| **LREG2** | Holds the 32-bit integer probability threshold (loaded once before the loop, reused every iteration) |
| **LREG3** | Holds the PRNG output; sign bit cleared for unsigned comparison |
| **LCONST_0** | Hardware constant 0.0; used as the zero value for dropped elements |
| **DEST** | Destination register file; holds tile data. Each iteration processes one row-pair (32 elements). Pointer auto-increments via addr_mod 3. |
| **Lane Flags** | Per-lane boolean flags set by `SFPIADD`; used by subsequent `SFPMOV` for conditional execution |
| **PRNG_SEED config reg** | Seeded once during `_init_dropout_` via `init_prng_seed` |

#### SFPU Execution Flow

1. **Initialization** (`_init_dropout_`): The PRNG seed is written to the `PRNG_SEED` configuration register. A 600-NOP wait ensures the seed has propagated to the PRNG hardware before any random numbers are generated.

2. **Constant Loading**: Before the main loop, the scale factor and probability threshold are loaded into LREG1 and LREG2 respectively. Each 32-bit value is loaded in two 16-bit halves using `SFPLOADI`.

3. **Per-iteration Processing** (8 iterations per tile, one per row-pair of 32 elements):
   - **Load from DEST**: `SFPLOAD` reads 32 elements from the current DEST position into LREG0.
   - **Scale**: `SFPMUL` multiplies LREG0 by LREG1 (scale factor). The result stays in LREG0.
   - **Generate random number**: `SFPMOV` in PRNG mode generates a per-lane pseudorandom uint32 into LREG3.
   - **Clear sign bit**: `SFPSETSGN` clears the sign bit of LREG3 so it can be compared as a non-negative integer against the probability.
   - **Compare and set flags**: `SFPIADD` computes `probability - random`. If the result is non-negative (probability >= random), the lane flag is set, meaning the element is KEPT. If negative (random > probability), the flag is not set, meaning the element is DROPPED.
   - **Conditional zero**: `SFPMOV` with lane flags active moves 0.0 into LREG0 for lanes where the flag is NOT set (dropped elements). Lanes where the flag IS set retain their scaled value in LREG0.
   - **Disable conditional mode**: `SFPENCC` clears the lane flag predication so subsequent instructions execute unconditionally.
   - **Store to DEST**: `SFPSTORE` writes LREG0 back to DEST. The auto-increment addressing moves the DEST pointer forward.
   - **Advance**: `dst_reg++` increments the C++ abstraction's DEST pointer for the next iteration.

4. **Tile lifecycle** (in the compute kernel):
   - `cb_wait_front(c_0, 1)`: Wait for reader to produce a tile.
   - `copy_tile(c_0, 0, 0)`: Unpack tile from CB c_0 into DEST[0].
   - `dropout_tile(0, prob, scale)`: Execute the SFPU dropout microcode on DEST[0] (8 iterations covering all 32x32 elements as 8 row-pairs of 32).
   - `pack_tile(0, c_2)`: Pack DEST[0] into output CB c_2.
   - `cb_pop_front(c_0, 1)`: Free the input tile.

#### SFPU Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Math Fidelity** | HiFi4 | Set in `ComputeConfig`; highest fidelity mode |
| **FP32 Dest Acc** | false | DEST accumulation uses default (BF16/FP16) precision |
| **Math Approx Mode** | false | No approximation; exact SFPU computation |
| **SFPU Type** | `SfpuType::dropout` | Used in `SFPU_ONE_PARAM_KERNEL_INIT` macro for init dispatch |
| **Vector Mode** | RC (Row-Column) | Processes all rows and columns of the tile face |
| **ITERATIONS** | 8 (default) | 8 iterations per tile = 8 row-pairs x 32 elements = 256 elements per face; two faces per tile half |

#### Hardware Compatibility Notes
The Wormhole B0 and Blackhole implementations of `_calculate_dropout_` are **identical**. Both architectures:
- Use the same SFPU instruction sequence
- Use the same PRNG mechanism (`SFPMOV` with `lreg_c=9, instr_mod1=8`)
- Have the same `init_prng_seed` function with 600-NOP wait
- The only difference in the Blackhole header is the additional `#include "sfpi_fp16.h"`, which does not affect dropout behavior

## Implementation Notes

1. **Inverted Dropout**: The operation implements inverted dropout -- elements that survive are scaled by `1/(1-p)` rather than scaling at inference time. The scale factor is precomputed on the host and passed as a bit-cast float.

2. **Probability Encoding**: The probability is converted from float to integer space as `(double)INT_MAX * prob`. This allows integer comparison in the SFPU (via `SFPIADD`) which is faster than floating-point comparison.

3. **PRNG Quality**: The hardware PRNG has "poor statistical properties" (per ISA documentation). This is acceptable for dropout where exact randomness is not critical, only approximate independence.

4. **Per-Device Seed**: The `DropoutMeshWorkloadFactory` variant offsets the seed by device ID (`seed += device->id()`) to ensure different dropout masks across mesh devices. This creates separate programs per device.

5. **Program Caching**: The `compute_program_hash` excludes the seed since it is a runtime argument. The probability and scale are compile-time arguments, so changing them invalidates the program cache.

6. **Conditional Execution Pattern**: The SFPU uses a two-step conditional pattern: (a) `SFPIADD` sets lane flags, (b) `SFPMOV` conditionally writes zeros, (c) `SFPENCC` disables conditional mode. This avoids branches -- all lanes execute all instructions, but only flagged lanes have their results overwritten.

7. **No `BACKWARDS` or `OUT_SHARDED` defines**: The program factory does not define these preprocessor flags, so the reader and writer kernels always use the forward sequential interleaved path.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the dropout operation work in TTNN? What program factory does it use, what kernels does it invoke, and how is the SFPU compute kernel structured for dropout?"
   **Reason**: Initial reconnaissance to understand the operation's architecture and locate all relevant files.
   **Key Findings**: Identified two program factory variants (DropoutProgramFactory and DropoutMeshWorkloadFactory), the kernel file paths, and the high-level structure of the SFPU kernel (dropout_tile, dropout_kernel_init).

2. **Query**: "Where is the dropout_tile SFPU function implemented? What file contains the actual SFPU kernel for dropout?"
   **Reason**: Needed to trace from the compute API (`dropout.h`) through the LLK layer to the actual SFPU microcode.
   **Key Findings**: Located the call chain: `dropout.h` -> `ckernel_sfpu_dropout.h` (arch-specific wrapper) -> `sfpu/ckernel_sfpu_dropout.h` (shared implementation in tt_llk submodule). The SFPU kernel uses `_calculate_dropout_` with raw TTI instructions.

3. **Query**: "What do these SFPU instructions do: SFPLOADI, SFPLOAD, SFPMUL, SFPMOV (PRNG), SFPSETSGN, SFPIADD, SFPENCC, SFPSTORE?"
   **Reason**: Needed authoritative descriptions of each SFPU instruction used in the dropout kernel.
   **Key Findings**: Obtained detailed descriptions of each instruction's operands, modes, and behavior. Confirmed SFPMOV with lreg_c=9/instr_mod1=8 triggers PRNG, SFPIADD with instr_mod1=10 sets lane flags for conditional execution, and SFPENCC controls the lane flag predication mask.

4. **Query**: "What does split_work_to_cores do in tt-metal?"
   **Reason**: Needed to understand the core distribution strategy used by the program factory.
   **Key Findings**: Returns a 6-tuple with num_cores, all_cores, core_group_1, core_group_2, and per-group tile counts. Group 1 gets ceil(tiles/cores), group 2 gets floor(tiles/cores).

### Documentation References

1. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel.h`
   **Reason**: Needed to understand `init_prng_seed` implementation.
   **Key Information**: Seeds the PRNG by writing to `PRNG_SEED_Seed_Val_ADDR32` config register, then waits 600 NOPs for propagation.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to understand the `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` and `SFPU_ONE_PARAM_KERNEL_INIT` macros.
   **Key Information**: `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` dispatches to `_llk_math_eltwise_unary_sfpu_params_` with the `calculate_dropout` function pointer, DST index, vector mode, and extra args (probability, scale). `SFPU_ONE_PARAM_KERNEL_INIT` calls `llk_math_eltwise_unary_sfpu_init` with the init callback and seed parameter.
