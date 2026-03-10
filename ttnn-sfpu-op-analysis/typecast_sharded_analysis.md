# Typecast (Sharded) Implementation Analysis

## Overview

The typecast sharded operation performs element-wise data type conversion on sharded tensors that reside entirely in L1 memory. Because input and output are both sharded, there is no DRAM involvement and no writer kernel -- the reader simply makes the sharded input visible to the compute kernel by pushing all tiles in a single `cb_push_back`, while the output circular buffer is directly backed by the sharded output tensor's L1 allocation.

**Program factory path**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp`

This factory is selected when `can_use_sharded_optimized_factory` is true, requiring: input is sharded, input/output tile sizes match, both buffers are in L1, and shard dimensions satisfy alignment constraints.

## Work Unit Definition

One work unit is **one tile** (32x32 elements). The compute kernel processes tiles one at a time in a single block: it iterates `num_tile_per_core` times, performing copy-to-DEST, SFPU typecast, and pack for each tile.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| Rank | Any (flattened to shard shape) |
| Dimension Convention | Last two dims form shard shape |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Sharded (L1) |
| Buffer Type | L1 |
| Data Type | Any supported input dtype (Float32, Float16_b, Bfp8_b, Bfp4_b, Int32, UInt32, UInt16, UInt8) |
| Shard Shape | `shard_spec.shape[0]` x `shard_spec.shape[1]` |
| Core Grid | Determined by `shard_spec.grid` |
| Shard Orientation | Determined by shard spec |

### Output Tensor

| Property | Value |
|---|---|
| Rank | Same as input |
| Dimension Convention | Same as input |
| Tensor Layout | TILE (32x32) |
| Memory Layout | Sharded (L1) |
| Buffer Type | L1 |
| Data Type | Target output dtype |
| Shard Shape | Same core count as input |
| Core Grid | Same as input |
| Shard Orientation | Same as input |

### Layout Transformations

No layout transformations occur. The operation converts data types in-place within tiles. The unpacker reads from the input CB in the input data format, data passes through DEST registers (optionally in FP32 mode), and the packer writes to the output CB in the output data format. For certain format pairs (e.g., Bfp8_b to Float16_b), no SFPU kernel executes at all -- the unpacker/packer hardware handles the conversion natively.

## Data Flow Pattern

1. **Reader kernel** (`reader_unary_sharded.cpp`): Executes `cb_push_back(cb_id_in0, num_tile_per_core)` to make all sharded input tiles visible to the compute kernel. No actual data movement occurs -- the CB is backed by the sharded tensor's L1 allocation via `set_globally_allocated_address`.

2. **Compute kernel** (`eltwise_typecast.cpp`): For each tile:
   - `cb_wait_front(input_cb, 1)` -- waits for one tile to be available
   - `copy_tile(input_cb, 0, 0)` -- unpacks tile from input CB into DEST register 0
   - `TYPECAST_LLK_INIT()` -- configures SFPU for the specific type conversion (called per tile, though init is typically idempotent)
   - `TYPECAST_LLK(0)` -- executes SFPU typecast on DEST[0]
   - `pack_tile(0, output_cb)` -- packs DEST[0] into output CB in the target data format
   - `cb_pop_front(input_cb, 1)` -- releases the consumed input tile

3. **No writer kernel**: The output CB is backed by the sharded output tensor's L1 allocation. The `cb_push_back(output_cb, per_core_block_dim)` at the end of the block makes all output tiles available.

## Circular Buffer Configuration

| CB ID | Name | Data Format | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| `c_0` | Input CB | Input dtype format | `round_up_to_mul32(tile_size(input_df))` | `num_tile_per_core` | page_size * num_pages | Single (1x) | Reader (sharded push) | Compute |
| `c_2` | Output CB | Output dtype format | `round_up_to_mul32(tile_size(output_df))` | `num_tile_per_core` | page_size * num_pages | Single (1x) | Compute | Host readback (sharded) |

Both CBs use `set_globally_allocated_address` to alias the sharded tensor buffers directly. The buffering factor is 1 because the data is already fully resident in L1.

## Pipeline Pattern Summary

Both CBs are single-buffered (buffering_factor = 1). Since data is already in L1 (sharded), there is no overlap between data movement and compute -- the reader instantly makes all tiles available, and the compute kernel processes them sequentially. This is a **fully sequential** pattern with no pipelining.

## Index Calculations

No complex index calculations are performed. Tile count per core is computed as:
- For **Bfp8_b / Bfp4_b** inputs: `ceil(shard_width / TILE_WIDTH) * ceil(shard_height / TILE_HEIGHT)`
- For **other formats**: `ceil(shard_height * shard_width * datum_size / tile_size)`

The compute kernel simply iterates tile index 0 through `num_tile_per_core - 1`, consuming and producing one tile at a time from fixed CB positions.

## Memory Access Patterns

### Read Pattern
Sequential tile-by-tile reads from the sharded L1 buffer through the input circular buffer. Each `cb_wait_front` + `copy_tile` reads one tile. The L1 address auto-advances as tiles are popped.

### Write Pattern
Sequential tile-by-tile writes to the sharded L1 output buffer through the output circular buffer. Each `pack_tile` writes one tile. All tiles are pushed in a single batch at the end of the block.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Defined by `shard_spec.grid` (the input tensor's shard specification) |
| Work Splitting | Each core processes exactly its own shard (`num_tile_per_core` tiles) |
| Load Balancing | Uniform -- all cores have the same shard size |
| Remainder Handling | None -- shard spec guarantees equal distribution |

The program factory uses `all_cores = shard_spec.grid` for kernel placement. Every core in the shard grid runs the same program with the same compile-time tile count.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `cb_id_in0` | `uint32_t` | Input circular buffer ID (always `c_0`) |

#### Compute Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `per_core_block_cnt` | `uint32_t` | Number of blocks per core (always 1) |
| 1 | `per_core_block_dim` | `uint32_t` | Tiles per block (`num_tile_per_core`) |
| 2 | `input_cb` | `uint32_t` | Input CB ID (`c_0`) |
| 3 | `output_cb` | `uint32_t` | Output CB ID (`c_2`) |

#### Compute Kernel Defines

| Define | Value | Description |
|---|---|---|
| `TYPECAST_LLK_INIT` | `typecast_tile_init<IN_FMT, OUT_FMT>` | SFPU init function with input/output DataFormat as template args |
| `TYPECAST_LLK` | `typecast_tile<IN_FMT, OUT_FMT>` | SFPU typecast function with input/output DataFormat as template args |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | `num_tile_per_core` | `uint32_t` | Number of tiles to push from sharded input |

## Kernel Implementations

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`
- **Key Logic**: This is the standard sharded reader -- a single `cb_push_back(cb_id_in0, num_tiles_per_core)` call. Since the CB is backed by the sharded tensor's L1 address, no data movement occurs. The push simply makes all tiles visible to the compute kernel's `cb_wait_front`.

### Compute Kernel

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // always 1 for sharded
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // num_tile_per_core
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);            // CB c_0
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);           // CB c_2

    init_sfpu(input_cb, output_cb);  // initializes unpack/pack pipelines for the given CB pair
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(output_cb, per_core_block_dim);  // reserve space for all output tiles at once
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for math RISC

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(input_cb, 1);  // wait for one input tile to be available

            copy_tile(input_cb, 0, 0);  // unpack tile 0 from input_cb into DEST[0]

            TYPECAST_LLK_INIT();  // configure SFPU for the specific typecast direction
            TYPECAST_LLK(0);      // execute SFPU typecast on DEST[0]

            tile_regs_commit();  // signal that math RISC is done writing DEST -- hand off to pack

            tile_regs_wait();  // wait for pack RISC to be ready to consume DEST

            pack_tile(0, output_cb);  // pack DEST[0] into output_cb in the target data format

            cb_pop_front(input_cb, 1);  // release consumed input tile from CB

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(output_cb, per_core_block_dim);  // publish all output tiles at once
    }
}
```

### SFPU Kernel Implementation

The typecast operation is unique among SFPU operations: it is not a single SFPU kernel but a **dispatch table** of ~30+ specialized conversion routines, selected at compile time based on the `IN_DTYPE` and `OUT_DTYPE` template parameters. Some type pairs require no SFPU kernel at all (handled purely by unpacker/packer hardware).

#### SFPU Kernel File
- **API layer**: `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`
- **LLK dispatch**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`
- **SFPU implementations (arch-specific wrappers)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`
- **SFPU implementations (shared core)**: `sfpu/ckernel_sfpu_typecast.h` (in tt_llk submodule, not checked out in this worktree)

#### Annotated SFPU Kernel Source (Blackhole arch-specific wrappers)

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_typecast.h"  // shared implementations from tt_llk submodule

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// --- FP32/FP16b -> UInt16 ---
// Delegates to shared implementation using SFPLOADMACRO pipeline
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint16() {
    _calculate_typecast_fp32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

// --- FP32/FP16b -> UInt8 (Blackhole-specific, inlined) ---
// Extracts exponent and mantissa, shifts to integer, handles sign via two's complement,
// masks to 8-bit range
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);  // load FP32 value from DEST into LREG0
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);    // extract biased exponent into LREG2
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);    // extract 8-bit mantissa into LREG1
        // shift amount = exponent - 23 (IEEE754 mantissa is 23 bits)
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);     // mantissa >>= (23 - exponent) to get integer part
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // enable lanes where input < 0
        // negate (two's complement) for negative inputs
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);  // add 256 bias
        TTI_SFPENCC(0, 0, 0, 0);  // re-enable all lanes
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);  // mask to 0xFF (LREG12 = vConstIntPrgm0 = 0xFF)
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_6, 0);  // store result back to DEST as int32
    }
}

// --- UInt{16,32} -> UInt8 (Blackhole-specific, inlined) ---
// Loads integer, adds 256 bias, masks to 8-bit
template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_7, 0);  // load lower 16 bits
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);  // load full 32-bit int
        }
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);  // add 256 bias
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);  // mask to 0xFF
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);  // store as int32
    }
}

// All other conversions delegate to shared tt_llk implementations:
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() { _calculate_typecast_uint16_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b() { _calculate_typecast_int32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_int32() { _calculate_typecast_fp32_to_int32_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() { _calculate_typecast_fp32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32() { _calculate_typecast_uint16_to_fp32_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() { _calculate_typecast_int32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32() { _calculate_typecast_fp32_to_uint32_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b() { _calculate_typecast_uint32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() { _calculate_typecast_uint32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32() { _calculate_typecast_uint16_to_uint32_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() { _calculate_typecast_uint32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>(); }
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() { _calculate_typecast_int32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>(); }

// --- Init functions ---
// Most delegate to shared tt_llk implementations that configure SFPU instruction templates
// and SFPCONFIG registers for the specific conversion direction.
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() { _init_typecast_fp32_to_fp16b_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() { _init_typecast_uint16_to_uint32_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() { _init_typecast_uint32_to_fp32_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp32() { _init_typecast_int32_to_fp32_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp32() { _init_typecast_uint16_to_fp32_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp16b() { _init_typecast_uint16_to_fp16b_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp16b() { _init_typecast_int32_to_fp16b_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp16b() { _init_typecast_uint32_to_fp16b_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint16() { _init_typecast_fp32_to_uint16_<APPROXIMATION_MODE>(); }

// Blackhole-specific: UInt8 init just sets the mask constant
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF;  // sets LREG12 to 0xFF for the AND mask in calculate_typecast_fp32_to_uint8
}
template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF;  // same mask constant for uint-to-uint8 conversion
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() { _init_typecast_uint32_to_uint16_<APPROXIMATION_MODE>(); }
template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() { _init_typecast_int32_to_uint16_<APPROXIMATION_MODE>(); }

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

The typecast operation uses a wide variety of SFPU instructions depending on the specific conversion. Key instructions across all conversion paths:

| Instruction | Description |
|---|---|
| `TTI_SFPLOAD` | Loads data from DEST register into an SFPU local register (LREG). Supports modes: DEFAULT (FP32), INT32, LO16 (lower 16 bits). |
| `TTI_SFPSTORE` | Stores data from SFPU local register back to DEST. Supports modes: INT32, LO16. |
| `TTI_SFPEXEXP` | Extracts the biased exponent from an IEEE 754 floating-point value. |
| `TTI_SFPEXMAN` | Extracts the mantissa (8-bit) from an IEEE 754 floating-point value. |
| `TTI_SFPIADD` | Integer add with immediate or register operand. Supports condition code modes and two's complement negation. |
| `TTI_SFPSHFT` | Arithmetic/logical shift by register-specified amount. |
| `TTI_SFPSETCC` | Sets per-lane condition codes based on register comparison (e.g., LT0 = less than zero). |
| `TTI_SFPENCC` | Enables all lanes (clears condition code mask). |
| `TTI_SFPAND` | Bitwise AND between two SFPU registers. |
| `TTI_SFPLOADI` | Loads an immediate value into an SFPU register (used for constants and instruction template configuration). |
| `TTI_SFPCONFIG` | Configures SFPU internal state: store modes, macro templates, load modes. |
| `TTI_SFPSWAP` | Conditional swap / max operation (used in fp32-to-uint16 init for clamping to non-negative). |
| `TTI_SFP_STOCH_RND` | Stochastic rounding instruction (used in fp32-to-uint16 for rounding during conversion). |
| `TTI_SFPCAST` | Type cast instruction for integer/float conversion (used in int32-to-fp32 paths). |
| `TTI_SFPMAD` | Multiply-add operation (used in int32/uint32 to float paths for sign reconstruction). |
| `TTI_SFPSHFT2` | Shift with immediate (used in fp32-to-fp16b for exponent/mantissa manipulation). |
| `TT_SFPLOADMACRO` | Macro-based load that pipelines load/simple/mad/round/store stages for high throughput (used in fp32-to-uint16). |
| `TTI_SFPNOP` | No-operation, used to pad pipeline stages in macro sequences. |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `LREG0` | Primary input value loaded from DEST |
| `LREG1` | Working register for mantissa / intermediate result / output |
| `LREG2` | Working register for exponent / shift amount |
| `LREG12` (`vConstIntPrgm0`) | Programmable constant (e.g., 0xFF mask for uint8 conversions) |
| `LREG13` (`vConstIntPrgm1`) | Programmable constant (e.g., 0x7FFF for fp16b rounding) |
| `LCONST_0` | Hardware constant register (value 0), used in two's complement negation |
| DEST[0] | Source/destination register for tile data -- `copy_tile` loads here, `pack_tile` reads from here |

#### SFPU Execution Flow

1. **Tile acquisition**: `cb_wait_front(input_cb, 1)` blocks until the reader has made a tile available.
2. **Unpack to DEST**: `copy_tile(input_cb, 0, 0)` triggers the unpacker to decode the input tile from its source data format and write it into DEST register 0. If `UnpackToDestMode::UnpackToDestFp32` is set (when `preserve_fp32_precision` is true), the unpacker writes full FP32 values to DEST.
3. **SFPU init**: `TYPECAST_LLK_INIT()` expands to `typecast_tile_init<IN_FMT, OUT_FMT>()` which calls the appropriate `init_typecast_*` function. This configures SFPU instruction templates, SFPCONFIG store modes, and programmable constants (e.g., `vConstIntPrgm0`).
4. **SFPU compute**: `TYPECAST_LLK(0)` expands to `typecast_tile<IN_FMT, OUT_FMT>(0)` which dispatches via `_llk_math_eltwise_unary_sfpu_params_` to the appropriate `calculate_typecast_*` function. The function iterates 8 times (ITERATIONS=8 for a tile face of 8 rows), processing one row per iteration. Each iteration loads from DEST, performs the type conversion via SFPU instructions, and stores back to DEST.
5. **Pack**: `pack_tile(0, output_cb)` triggers the packer to read DEST[0] and encode into the output data format, writing into the output CB.
6. **Release**: `cb_pop_front` frees the input tile; `tile_regs_release` frees DEST for the next tile.

For some type pairs (Bfp8_b <-> Float16_b, Bfp4_b <-> Float16_b, Bfp8_b <-> Float32, etc.), the SFPU body is empty -- the conversion is handled entirely by the unpacker (which decodes block floating point to FP16b/FP32) and/or the packer (which encodes to block floating point format).

#### SFPU Configuration

| Setting | Value | Source |
|---|---|---|
| Math Fidelity | `HiFi4` | Hardcoded in program factory |
| Math Approx Mode | `false` | Hardcoded (`APPROXIMATION_MODE = false`) |
| FP32 Dest Acc | Configurable | `args.fp32_dest_acc_en` -- enables 32-bit DEST accumulator |
| Unpack to Dest Mode | Configurable | `UnpackToDestFp32` when `preserve_fp32_precision` is true |
| Bfp8 Pack Precise | Configurable | `args.bfp8_pack_precise` -- higher precision for Bfp8_b packing |
| SfpuType | `typecast` | Used in `llk_math_eltwise_unary_sfpu_init` |

#### Hardware Compatibility Notes

- The Blackhole and Wormhole implementations are structurally identical. The only difference is in address modifier constants used for `TTI_SFPLOAD`/`TTI_SFPSTORE` in the arch-specific `calculate_typecast_fp32_to_uint8` and `calculate_typecast_uint_to_uint8` functions:
  - **Blackhole**: Uses `ADDR_MOD_7` for load and `ADDR_MOD_6` for store.
  - **Wormhole**: Uses `ADDR_MOD_3` for load and `ADDR_MOD_2` for store.
- All other conversion functions delegate to the shared `sfpu/ckernel_sfpu_typecast.h` in the tt_llk submodule and are identical across architectures.
- The SFPLOADMACRO-based conversions (fp32-to-uint16) use pipelined execution for 2-cycle-per-row throughput.
- The `TTI_SFP_STOCH_RND` instruction (used in fp32-to-uint16 init) may have different rounding behavior modes between architectures but the API is the same.

## Implementation Notes

1. **No writer kernel**: This is a distinctive feature of the sharded variant. Because both input and output CBs are aliased to sharded L1 buffers via `set_globally_allocated_address`, no explicit data movement kernels are needed for output.

2. **Tile size constraint**: The factory asserts `input_tile_size == output_tile_size`. This limits the sharded factory to conversions where both formats have the same encoded tile size, which rules out conversions between formats with different tile sizes (e.g., Float32 to Bfp4_b).

3. **TYPECAST_LLK_INIT called per tile**: The init function is called inside the per-tile loop rather than once before it. For most conversion types, the init is idempotent (just sets constants/config), so this is safe but slightly redundant.

4. **Large dispatch table**: The LLK layer (`llk_math_eltwise_unary_sfpu_typecast.h`) contains a massive `if constexpr` chain that selects the appropriate SFPU kernel at compile time based on the IN_DTYPE/OUT_DTYPE template parameters. This means only the relevant conversion code is compiled into the kernel binary.

5. **Packer/Unpacker-only conversions**: Several type pairs (Bfp8_b <-> Float16_b, Bfp4_b <-> Float16_b, etc.) require no SFPU math at all. The unpacker decodes block floating point to the DEST register format, and the packer encodes from DEST to the target format. The SFPU typecast call body is empty for these cases.

6. **Program caching**: The `override_runtime_arguments` method only updates the globally-allocated CB addresses via `UpdateDynamicCircularBufferAddress`, enabling efficient reuse of the compiled program when tensor addresses change but shapes/types remain the same.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the typecast sharded program factory work? What kernels does it use and how does it handle sharded tensors for type conversion?"
   **Reason**: Initial understanding of the sharded typecast factory architecture and kernel selection.
   **Key Findings**: Confirmed that the sharded factory is selected when input is sharded with matching tile sizes, both buffers in L1. Identified preconditions and fallback to general TypecastProgramFactory.

2. **Query**: "How does typecast_tile_init and typecast_tile work in the LLK layer? What SFPU operations do they perform for data type conversion? Where is the implementation?"
   **Reason**: Understanding the SFPU kernel layer that the compute kernel dispatches to.
   **Key Findings**: Identified the `_calculate_typecast_*` and `_init_typecast_*` functions in `ckernel_sfpu_typecast.h`, the use of SFPLOADMACRO for pipelined conversions, and the wide variety of TTI_SFP* instructions used across conversion types.

3. **Query**: "Show the full source code of _calculate_typecast_fp32_to_uint16_, _calculate_typecast_fp32_to_int32_, _init_typecast_fp32_to_uint16_, and _init_typecast_fp32_to_fp16b_"
   **Reason**: Obtaining the actual SFPU instruction sequences for key conversion paths (from the tt_llk submodule which was not checked out locally).
   **Key Findings**: Retrieved full implementations showing SFPLOADMACRO pipelining for fp32->uint16, manual exponent/mantissa extraction for fp32->int32, SFPCONFIG-based instruction template setup for init functions, and SFPSHFT2-based fp32->fp16b conversion.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op_types.hpp`
   **Reason**: Understanding the TypecastParams and TypecastInputs structures.
   **Key Information**: Documented the fp32_dest_acc_en, preserve_fp32_precision, and bfp8_pack_precise configuration flags.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`
   **Reason**: Understanding the compute API layer that maps typecast_tile/typecast_tile_init to LLK calls.
   **Key Information**: Confirmed the template parameter passing pattern and the extensive list of supported type conversion pairs documented in the API comments.
