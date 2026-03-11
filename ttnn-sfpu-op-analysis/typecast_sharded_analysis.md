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

## Implementation Notes

1. **No writer kernel**: This is a distinctive feature of the sharded variant. Because both input and output CBs are aliased to sharded L1 buffers via `set_globally_allocated_address`, no explicit data movement kernels are needed for output.

2. **Tile size constraint**: The factory asserts `input_tile_size == output_tile_size`. This limits the sharded factory to conversions where both formats have the same encoded tile size, which rules out conversions between formats with different tile sizes (e.g., Float32 to Bfp4_b).

3. **TYPECAST_LLK_INIT called per tile**: The init function is called inside the per-tile loop rather than once before it. For most conversion types, the init is idempotent (just sets constants/config), so this is safe but slightly redundant.

4. **Large dispatch table**: The LLK layer (`llk_math_eltwise_unary_sfpu_typecast.h`) contains a massive `if constexpr` chain that selects the appropriate SFPU kernel at compile time based on the IN_DTYPE/OUT_DTYPE template parameters. This means only the relevant conversion code is compiled into the kernel binary.

5. **Packer/Unpacker-only conversions**: Several type pairs (Bfp8_b <-> Float16_b, Bfp4_b <-> Float16_b, etc.) require no SFPU math at all. The unpacker decodes block floating point to the DEST register format, and the packer encodes from DEST to the target format. The SFPU typecast call body is empty for these cases.

6. **Program caching**: The `override_runtime_arguments` method only updates the globally-allocated CB addresses via `UpdateDynamicCircularBufferAddress`, enabling efficient reuse of the compiled program when tensor addresses change but shapes/types remain the same.

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. The typecast operation is unique: it is not a single SFPU kernel but a **dispatch table of ~30+ specialized conversion routines**, selected at compile time based on `IN_DTYPE` and `OUT_DTYPE` template parameters. Some type pairs require no SFPU kernel at all and are handled purely by the unpacker/packer hardware.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_typecast.h` (shared core), with arch-specific wrappers at `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `TYPECAST_LLK(0)` which expands (via preprocessor define) to `typecast_tile<IN_FMT, OUT_FMT>(0)`.
2. `typecast_tile` (in `typecast.h`) invokes `llk_math_eltwise_unary_sfpu_typecast<APPROX, IN_DTYPE, OUT_DTYPE>(idst)` wrapped in the `MATH()` macro (runs on math RISC only).
3. `llk_math_eltwise_unary_sfpu_typecast` (in `llk_math_eltwise_unary_sfpu_typecast.h`) uses a large `if constexpr` chain to select the appropriate conversion function and calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_typecast_*, dst_index, vector_mode)`.
4. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, configures addr_mod base, stalls until SFPU is ready, then calls the SFPU function 4 times (for VectorMode::RC, once per face), advancing the DEST pointer by 16 rows between faces.
5. The SFPU function (e.g., `calculate_typecast_fp32_to_uint16`) either directly contains TTI instructions or delegates to the shared `_calculate_typecast_*_` implementation in the tt_llk submodule.

Similarly, `TYPECAST_LLK_INIT()` expands to `typecast_tile_init<IN_FMT, OUT_FMT>()`, which calls `llk_math_eltwise_unary_sfpu_typecast_init`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::typecast, APPROXIMATE>(init_func)`. The generic init calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::typecast>()` (which configures SFPU config registers and addr_mod) and then the conversion-specific init function (which sets up instruction templates, programmable constants, and SFPCONFIG store modes).

### Annotated SFPU Kernel Source

The typecast operation has many conversion paths. Below is the complete core SFPU implementation from the Wormhole B0 tt_llk submodule (the Blackhole variant differs only in `ADDR_MOD` indices for the non-SFPLOADMACRO paths). The arch-specific wrapper functions (for `fp32_to_uint8` and `uint_to_uint8`) are also included since they are inlined at the wrapper level and not delegated to the shared implementation.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 2-cycle-per-row throughput.
    // Pipeline: Load -> Simple(max(v,0.0)) -> Round(SFPSTOCHRND to uint16) -> Store(L16)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between LREG0 and LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, v >> 2);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 1-cycle-per-row throughput.
    // Pipeline: Load(LO16) -> Simple(cast) -> Round(SFPSTOCHRND fp32->fp16b) -> Store(L16)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate between LREG0 and LREG1
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 4-cycle-per-row throughput.
    // L0=0.0, L1=-2^31. Sign bit stored in L7 for indirect VA in SFPMAD.
    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG: L7 = t >> 31
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
    // SFPI implementation: converts IEEE 754 FP32 to two's complement INT32.
    // Algorithm: extract exponent/mantissa, shift to integer, handle overflow/underflow, apply sign.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];

        sfpi::vInt exp = sfpi::exexp(in);       // extract debiased exponent
        sfpi::vUInt man = sfpi::exman8(in);      // extract mantissa with implicit 1 at bit 23
        sfpi::vInt shift_amt = exp - 23;         // shift amount for mantissa alignment
        sfpi::vInt result = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(man, shift_amt));

        v_if (exp >= 31) {
            result = 0x80000000;  // INT_MIN for overflow
        }
        v_endif;

        v_if (exp < 0) {
            result = 0;  // underflow: |value| < 1
        }
        v_endif;

        v_if (in < 0.0f) {
            result = ~result + 1;  // two's complement negation
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0); // load FP32 from DEST, no incr
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0); // result = 0

        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0); // LaneEnabled = in >= 0
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP); // exp; CC = exp >= 0
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff); // result = 0xffffffff (UINT_MAX)
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0); // exp -= 32; CC = exp < 32
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE); // exp += 9
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // mantissa with implicit 1
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // mantissa << (exp - 23)
        TTI_SFPENCC(0, 0, 0, 0); // re-enable all lanes

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0); // store, DEST incr +2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 3-cycle-per-row throughput.
    // Round-to-nearest-even using LSB extraction and 0x7fff bias add.
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // alternate between LREG0 and LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_3, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_2, b >> 2);
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 1-cycle-per-row throughput.
    // Pipeline: Load(LO16) -> Simple(cast to FP32) -> Store(L16, FP32)
    constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 4-cycle-per-row throughput.
    // L0=0.0, L1=-2^31. abs(v) -> cast -> setsgn -> SFPMAD with indirect VA using L7 sign bit.
    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00); // -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG: L7 = t >> 31
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 3-cycle-per-row throughput.
    // L0=0.0, L1=2^31. Sign bit in L7 for indirect VA.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate between LREG2 and LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5); // L7 = v >> 31
        TT_SFPSETSGN(0, v, v, 1); // clear sign bit (SFPSETSGN_MOD1_ARG_IMM)
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 3-cycle-per-row throughput.
    // L0=0.0, L1=2^31. Three macros: [a]=setsgn(a,0), [b]=cast(a), [L7]=shift>>31 + MAD + store.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2^31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_3, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_2, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 1-cycle-per-row throughput.
    // Pipeline: Load(LO16) -> Store(INT32)
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_2, 0);
    }
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 2-cycle-per-row throughput.
    // Loads high 16 bits, negates, shifts right by 16, then ORs with low-16-bit load to form truncated uint16.
    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 9
    for (int d = 0; d < ITERATIONS + 1; d++)
    {
        int a          = d & 1;
        int macroIndex = 1 + (d & 1);
        if (d < ITERATIONS)
        {
            TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::LO16, ADDR_MOD_2, a >> 2);
        }
        else
        {
            TTI_SFPNOP;
        }
        if (d == 0)
        {
            TTI_SFPNOP;
        }
        else if (d < ITERATIONS)
        {
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, (-4 & 0x3ff) | (b >> 2));
        }
        else
        {
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_2, (-2 & 0x3ff) | (b >> 2));
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 3-cycle-per-row throughput.
    // cast_fp32(a) -> max(0.0, a) -> SFPSTOCHRND to uint16 (clamps to 65535).
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, a >> 2);
        TT_SFPCAST(a, a, 0);
        TTI_SFPNOP;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

Arch-specific wrappers for `fp32_to_uint8` and `uint_to_uint8` (Wormhole B0 shown; Blackhole uses `ADDR_MOD_7`/`ADDR_MOD_6` instead of `ADDR_MOD_3`/`ADDR_MOD_2`):

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() { // APPROXIMATION_MODE=false, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0); // load FP32 from DEST, no auto-incr
        // exponent = exexp(in)
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // mantissa = exman8(in)
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // exponent -= 23  ->  shift amount = exponent - 23
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa = mantissa >> (23 - exponent)
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // LaneEnabled = in < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // mantissa = ~mantissa + 1  (two's complement)
        TTI_SFPIADD(
            0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // mantissa += 256
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // LaneEnabled = true
        TTI_SFPENCC(0, 0, 0, 0);
        // mantissa &= 0xFF (LREG12 = vConstIntPrgm0 = 0xFF)
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0); // store, DEST incr +2
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() { // APPROXIMATION_MODE=false, ITERATIONS=8
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0); // load lower 16 bits
        } else {
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0); // load full 32-bit int
        }
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0); // mask to 0xFF
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0); // store, DEST incr +2
    }
}
```

Key init functions (Wormhole B0 shared core shown):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_fp16b_()
{
    constexpr int b = p_sfpu::LREG2;

    sfpi::vConstIntPrgm0 = 1;       // LREG12 = 1 (LSB mask)
    sfpi::vConstIntPrgm1 = 0x7fff;  // LREG13 = 0x7FFF (round-to-nearest bias)

    // InstructionTemplate[0]: right-shift by 16 (extract upper 16 bits)
    TTI_SFPSHFT2(-16 & 0xfff, 0, 12, 6); // SFPSHFT2_MOD1_SHFT_IMM
    // InstructionTemplate[1]: add vConstIntPrgm1 (0x7FFF rounding bias)
    TTI_SFPIADD(0, p_sfpu::LREG13, 13, sfpi::SFPIADD_MOD1_CC_NONE);
    // InstructionTemplate[2]: add LREG2 (combines LSB + rounded value)
    TTI_SFPIADD(0, b, 14, sfpi::SFPIADD_MOD1_CC_NONE);

    // Macro 0: [a] - extracts LSB, schedules 32-bit store
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x40 | (3 << 3) | (4 + 2);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1: [b] - adds 0x7FFF bias, schedules 16-bit FP16B store
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (1 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    // StoreMod0=FP16B, UsesLoadMod0ForStore={1,0}, UnitDelayKind={1,1}
    TTI_SFPCONFIG(0x310 | InstrModLoadStore::FP16B, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_fp32_to_uint16_()
{
    // InstructionTemplate[0]: swap/max to clamp negative values to 0
    TTI_SFPSWAP(0, p_sfpu::LCONST_0, 12, 0xf); // L[VD] = max(0, L[VD])
    // InstructionTemplate[1]: stochastic round FP32 -> UINT16
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 13, 6); // SFPSTOCHRND_MOD1_FP32_TO_UINT16

    // Macro 0: Load -> max(0,v) -> SFPSTOCHRND(uint16) -> Store(LO16)
    {
        constexpr std::uint32_t simple_bits = 0x80 | 0x00 | (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0x00 | 0x40 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t store_bits  = 0x00 | 0x40 | (3 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // StoreMod0=LO16, UsesLoadMod0ForStore={0}, UnitDelayKind={1}
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::LO16, 8, 1);
}

template <bool APPROXIMATION_MODE>
inline void _init_typecast_uint16_to_uint32_()
{
    // Simplest init: just load and store, no transformation.
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (0 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // StoreMod0=INT32
    TTI_SFPCONFIG(0x100 | InstrModLoadStore::INT32, 8, 1);
}

// init_typecast_fp32_to_uint8 (both architectures, in arch-specific wrapper):
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF; // sets LREG12 to 0xFF for the AND mask
}

// init_typecast_uint_to_uint8 (both architectures, in arch-specific wrapper):
template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() {
    sfpi::vConstIntPrgm0 = 0xFF; // same 0xFF mask constant
}
```

### SFPU Instructions Used

The typecast operation uses a wide variety of SFPU instructions across its many conversion paths:

| Instruction | Description |
|-------------|-------------|
| `TTI_SFPLOAD` | Loads data from DEST register into an SFPU local register (LREG). Supports load modes: `DEFAULT` (FP32 reinterpret), `INT32` (integer), `LO16` (lower 16 bits zero-extended). |
| `TTI_SFPSTORE` | Stores data from SFPU local register back to DEST. Supports store modes: `INT32`, `LO16`, `FP32`, `FP16B`. |
| `TT_SFPLOADMACRO` / `TTI_SFPLOADMACRO` | Macro-based pipelined instruction that orchestrates Load, Simple, MAD, Round, and Store sub-units in a single dispatch. Achieves 1-4 cycle-per-row throughput depending on conversion complexity. The macro index selects a pre-configured instruction template set up during init. |
| `TTI_SFPEXEXP` | Extracts the debiased exponent from an IEEE 754 floating-point value. Optional CC modes: `SFPEXEXP_MOD1_SET_CC_SGN_EXP` sets CC based on exponent sign, `SFPEXEXP_MOD1_SET_CC_COMP_EXP` complements the CC. |
| `TTI_SFPEXMAN` | Extracts the mantissa (with implicit leading 1 at bit 23) from an IEEE 754 floating-point value. |
| `TTI_SFPIADD` | Integer add with immediate or register operand. Key modes: `SFPIADD_MOD1_ARG_IMM` (immediate operand), `SFPIADD_MOD1_ARG_2SCOMP_LREG_DST` (two's complement negate dest), `SFPIADD_MOD1_CC_NONE` (unconditional), `SFPIADD_MOD1_CC_LT0` (execute only if CC indicates < 0). |
| `TTI_SFPSHFT` | Shift by register-specified amount. Positive = left shift, negative = right shift. |
| `TTI_SFPSHFT2` | Shift with immediate or register operand. Mode 5 (`SFPSHFT2_MOD1_SHFT_LREG`) shifts by LREG value. Mode 6 (`SFPSHFT2_MOD1_SHFT_IMM`) shifts by immediate. |
| `TTI_SFPSETCC` | Sets per-lane condition codes. `SFPSETCC_MOD1_LREG_LT0` enables lanes where register < 0. `SFPSETCC_MOD1_LREG_GTE0` enables lanes where register >= 0. |
| `TTI_SFPENCC` | Enables all lanes unconditionally (clears condition code mask). |
| `TTI_SFPAND` | Bitwise AND between two SFPU registers. Used for bit masking (e.g., `& 0xFF` for uint8 truncation). |
| `TTI_SFPLOADI` | Loads an immediate value into an SFPU register. Modes: `SFPLOADI_MOD0_USHORT` (unsigned 16-bit), `SFPLOADI_MOD0_SHORT` (sign-extended 16-bit), `SFPLOADI_MOD0_FLOATB` (BFloat16 format), `SFPLOADI_MOD0_LOWER`/`SFPLOADI_MOD0_UPPER` (for SFPLOADMACRO config). |
| `TTI_SFPCONFIG` | Configures SFPU internal state: SFPLOADMACRO instruction templates, store format modes, delay kinds. Used extensively in init functions to set up the macro pipeline. |
| `TTI_SFPSWAP` | Conditional swap / min-max operation. With `mod1=0xf`, computes `L[VD] = max(0, L[VD])` -- used to clamp negatives before uint16 conversion. |
| `TTI_SFP_STOCH_RND` | Stochastic/deterministic rounding instruction. Mode 1 (`SFPSTOCHRND_MOD1_FP32_TO_FP16B`) rounds FP32 to FP16B. Mode 6 (`SFPSTOCHRND_MOD1_FP32_TO_UINT16`) converts FP32 to uint16 with clamping. |
| `TTI_SFPCAST` | Integer-to-float cast instruction. Converts unsigned integer in LREG to IEEE 754 FP32. |
| `TTI_SFPMAD` | Fused multiply-add: `result = VA * VB + VC`. Mode 4 (`SFPMAD_MOD1_INDIRECT_VA`) uses L7 as an index to select VA from L0/L1, enabling sign-dependent correction (e.g., adding -2^31 for negative int32 values). |
| `TTI_SFPABS` / `TT_SFPABS` | Computes absolute value of an SFPU register. |
| `TTI_SFPSETSGN` / `TT_SFPSETSGN` | Sets or clears the sign bit. Mode 0 copies sign from source register; mode 1 (`SFPSETSGN_MOD1_ARG_IMM`) clears sign bit (makes positive). |
| `TTI_SFPOR` | Bitwise OR between two SFPU registers. Used in uint32-to-uint16 to combine high/low 16-bit halves. |
| `TTI_SFPGT` | Greater-than comparison. With `SFPGT_MOD1_SET_VD`, sets VD to -1 (all ones) if value > 0, else 0. Used in Blackhole uint32-to-uint16 init. |
| `TTI_SFPNOP` | No-operation. Required as pipeline padding for SFPLOADMACRO sequences to allow pipeline stages to complete. |
| `TTI_STALLWAIT` | Stalls until the specified hardware unit is ready. Used in the params dispatch to synchronize SFPU with math pipeline. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | Primary working register; alternates with LREG1 in SFPLOADMACRO pipelining; holds loaded FP32/INT32 input; constant 0.0 in int32/uint32 conversion paths |
| `LREG1` | Secondary working register; alternates with LREG0 in SFPLOADMACRO pipelining; holds mantissa/intermediate result in TTI paths; constant -2^31 or 2^31 in int32/uint32 paths |
| `LREG2` | Working register for exponent/shift amount in TTI paths; alternates with LREG3 as `v` in SFPLOADMACRO int32 paths; used as `b` register in fp32_to_fp16b |
| `LREG3` | Alternates with LREG2 as `v` in SFPLOADMACRO int32/uint32 paths; used as `b` in uint32_to_fp32 |
| `LREG4` | Temporary register `t` in int32_to_fp16b and int32_to_fp32 paths (holds abs value during sign processing) |
| `LREG7` | Sign bit storage register. SFPLOADMACRO with `SFPMAD_MOD1_INDIRECT_VA` uses L7 as an index into L0/L1 to select the correction constant based on the sign bit of the input. |
| `LREG12` (`vConstIntPrgm0`) | Programmable constant register. Set to `0xFF` for uint8 mask, `1` for fp32_to_fp16b LSB extraction, `-31` for fp32/int32 conversion shift amounts. |
| `LREG13` (`vConstIntPrgm1`) | Programmable constant register. Set to `0x7FFF` for fp32_to_fp16b round-to-nearest-even bias. |
| `LCONST_0` | Hardware constant register (value 0.0). Used as the zero operand in two's complement negation (`SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`), and as the comparison value in `SFPSWAP` for clamping negatives to zero. |
| `LCONST_1` | Hardware constant register (value 1.0). Used as VB in `SFPMAD` for the identity multiply `L[L7] * 1.0 + v`. |
| DEST registers | Source and destination for tile data. `copy_tile` loads input here; the SFPU reads/writes via `SFPLOAD`/`SFPSTORE` or `SFPLOADMACRO`; `pack_tile` reads from here. |

### Address Mode Configuration

The typecast operation configures two address modes during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::typecast>()`:

**ADDR_MOD_7** (no auto-increment):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},  // DEST pointer does NOT advance after SFPLOAD
}
```
Used by `TTI_SFPLOAD` instructions in the non-SFPLOADMACRO paths (e.g., `fp32_to_uint8`, `fp32_to_uint32`, `fp32_to_int32`). These paths manually manage DEST addressing through the SFPLOADMACRO pipeline or the `SETRWC` instructions in the params dispatch.

**ADDR_MOD_6** (auto-increment by 2):
```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 2},  // DEST pointer advances by 2 rows after SFPSTORE
}
```
Used by `TTI_SFPSTORE` in the non-SFPLOADMACRO paths. The increment of 2 advances through the 16 rows of a tile face (8 iterations x 2 rows = 16 rows per face).

**Architecture differences**: The address mode configuration is **identical** between Wormhole B0 and Blackhole for the typecast operation. Both use `ADDR_MOD_7` (dest.incr=0) and `ADDR_MOD_6` (dest.incr=2). However, the **Wormhole B0 arch-specific wrapper** for `fp32_to_uint8` and `uint_to_uint8` uses `ADDR_MOD_3` for load and `ADDR_MOD_2` for store instead of `ADDR_MOD_7`/`ADDR_MOD_6`, because these are the default SFPU addr_mods with equivalent increments (ADDR_MOD_3: dest.incr=0, ADDR_MOD_2: dest.incr=2) that are set up by the A2D (unpack-to-DEST) infrastructure. The Blackhole arch-specific wrapper uses `ADDR_MOD_7`/`ADDR_MOD_6` consistently.

**SFPLOADMACRO paths**: The SFPLOADMACRO-based conversion functions use `ADDR_MOD_2` or `ADDR_MOD_3` as parameters to the macro instruction itself. The macro handles DEST address advancement internally as part of its pipelined scheduling, using the address mode's `dest.incr` field to step through rows.

**Face-to-face advancement**: After each SFPU function call (which processes 8 rows = one face half), the `_llk_math_eltwise_unary_sfpu_params_` dispatch advances the DEST pointer by 16 rows using two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` calls to move to the next face.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "Where is the typecast_tile SFPU kernel implementation? I'm looking for the compute API header for typecast (typecast.h), the LLK dispatch layer, and the core ckernel SFPU implementation (ckernel_sfpu_typecast.h or similar). What does the typecast SFPU operation do?"
   **Reason**: Initial research to locate all abstraction layers and understand the typecast SFPU architecture.
   **Key Findings**: Identified the three-layer structure: API header at `typecast.h`, LLK dispatch at `llk_math_eltwise_unary_sfpu_typecast.h`, and core implementations in `ckernel_sfpu_typecast.h` at both arch-specific wrapper and shared tt_llk levels. Confirmed that typecast supports 20+ conversion combinations and that some pairs (Bfp to Float16_b) require no SFPU kernel.

### Confluence References
No Confluence SFPU ISA page consultation was needed. The typecast operation's SFPU instructions (SFPLOAD, SFPSTORE, SFPEXEXP, SFPEXMAN, SFPIADD, SFPSHFT, SFPSETCC, SFPENCC, SFPAND, SFPLOADMACRO, SFPCAST, SFPMAD, SFPSWAP, SFP_STOCH_RND, SFPCONFIG) are well-documented in the source code comments and DeepWiki was sufficient.

### Glean References
No Glean consultation was needed for this analysis.
