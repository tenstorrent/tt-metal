# TYPECAST (Sharded) - SFPU Operation Analysis

## Operation Overview

**Operation Name**: typecast (sharded variant)
**Category**: copy / data format conversion
**Namespace**: `ttnn::prim`
**Program Factory**: `TypecastShardedProgramFactory`

The typecast operation converts tensor data between different numeric formats (e.g., Float16_b, Float32, UInt16, UInt32, Int32, Bfp8_b, Bfp4_b) using the SFPU vector unit. The sharded variant operates on tensors whose data already resides in L1 SRAM (sharded memory layout), avoiding DRAM reads/writes entirely. Both the input and output circular buffers are backed by globally-allocated L1 buffers, meaning the reader kernel merely "publishes" already-resident pages rather than fetching data over the NoC.

### Supported Type Conversions

The typecast operation supports a large matrix of conversions. Some are handled entirely by the SFPU, while others are handled by the unpacker/packer hardware without needing SFPU intervention:

**SFPU-driven conversions** (require `calculate_typecast_*` kernel):
- Float16_b <-> UInt16, Int32, UInt32
- Float32 <-> UInt16, Int32, UInt32, Float16_b
- Bfp8_b <-> UInt16, Int32, UInt32
- Bfp4_b <-> UInt16, Int32, UInt32
- UInt16 <-> UInt32, Int32, Float32, Float16_b, Bfp8_b, Bfp4_b
- Int32 <-> UInt16
- UInt32 <-> UInt16, Float32, Float16_b, Bfp8_b, Bfp4_b

**Packer/Unpacker-only conversions** (no SFPU kernel body; the `llk_math` function is a no-op):
- Float16_b <-> Float32 (packer handles precision change)
- Bfp8_b <-> Float16_b, Float32 (unpacker/packer)
- Bfp4_b <-> Float16_b, Bfp8_b, Float32 (unpacker/packer)

---

## Program Factory Analysis

### File
`ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp`

### Program Structure

| Component | Count | Details |
|-----------|-------|---------|
| Reader Kernels | 1 | Sharded reader (publishes existing L1 pages) |
| Compute Kernels | 1 | SFPU typecast (format-parameterized) |
| Writer Kernels | 0 | No writer -- output CB is globally allocated in L1 |
| Circular Buffers | 2 | c_0 (input, sharded), c_2 (output, sharded) |

### Execution Model
- **Core Grid**: Derived from the input tensor's shard spec (`shard_spec.grid`)
- **Parallelism**: SPMD across all sharded cores; each core processes `num_tile_per_core` tiles
- **No NoC traffic**: Both input and output reside in L1; the reader just calls `cb_push_back` to make data visible

### Operation Parameters (TypecastParams)

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dtype` | `DataType` | Source data format (used to parameterize SFPU kernel) |
| `output_dtype` | `DataType` | Destination data format (used to parameterize SFPU kernel) |
| `output_memory_config` | `MemoryConfig` | Memory configuration for the output tensor |
| `fp32_dest_acc_en` | `bool` | Enable 32-bit accumulation in DST registers |
| `preserve_fp32_precision` | `bool` | Use `UnpackToDestFp32` mode for input CB |
| `bfp8_pack_precise` | `bool` | Enable precise BFP8 packing |

### Compile-Time Defines

The program factory injects two critical defines into the compute kernel:

| Define | Value Pattern | Purpose |
|--------|---------------|---------|
| `TYPECAST_LLK_INIT` | `typecast_tile_init<IN_FMT, OUT_FMT>` | Initializes SFPU for the specific conversion |
| `TYPECAST_LLK` | `typecast_tile<IN_FMT, OUT_FMT>` | Executes the conversion on a single tile in DST |

The `IN_FMT` and `OUT_FMT` are the `uint32_t` casts of `tt::DataFormat` enum values, making the entire conversion path a compile-time decision via `if constexpr` chains.

---

## Circular Buffer Configuration

### CB c_0 (Input)
| Property | Value |
|----------|-------|
| CB Index | `tt::CBIndex::c_0` |
| Data Format | Input tensor's data format |
| Page Size | `round_up_to_mul32(tile_size(act_df))` |
| Num Pages | `num_tile_per_core` (all tiles for this core's shard) |
| Buffering | 1x (single-buffered; data already in L1) |
| Backing | Globally allocated to input tensor's L1 buffer |

### CB c_2 (Output)
| Property | Value |
|----------|-------|
| CB Index | `tt::CBIndex::c_2` |
| Data Format | Output tensor's data format |
| Page Size | `round_up_to_mul32(tile_size(out_df))` |
| Num Pages | `num_tile_per_core` |
| Buffering | 1x (single-buffered; data already in L1) |
| Backing | Globally allocated to output tensor's L1 buffer |

### Tile Count Calculation

For BFP formats (Bfp8_b, Bfp4_b):
```
num_tile_per_core = ceil(shard_width / TILE_WIDTH) * ceil(shard_height / TILE_HEIGHT)
```

For other formats:
```
shard_size_in_bytes = shard_height * shard_width * datum_size(act_df)
num_tile_per_core = ceil(shard_size_in_bytes / input_tile_size)
```

An important constraint: `input_tile_size == output_tile_size` is asserted. This means the sharded factory is only selected when source and destination tile sizes match, which the `can_use_sharded_optimized_factory` function verifies.

---

## Kernel Implementations

### Reader Kernel
**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp`

This is a minimal sharded reader that simply makes the already-resident L1 data visible to the compute kernel by pushing pages into the input circular buffer.

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"

void kernel_main() {
    // Runtime arg 0: number of tiles in this core's shard
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    // Compile-time arg 0: which CB to push pages into (c_0)
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    // Since this is a sharded tensor, the data is already resident in L1
    // at the address backing this CB. We just need to signal to the compute
    // kernel that all pages are available for reading.
    cb_push_back(cb_id_in0, num_tiles_per_core);
}
```

### Compute Kernel
**File**: `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"     // Provides typecast_tile<> and typecast_tile_init<>

void kernel_main() {
    // Compile-time args from program factory:
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Always 1 for sharded
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // num_tile_per_core
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);            // c_0
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);           // c_2

    // Initialize the SFPU for unary operation mode. This configures the
    // unpack-to-DST and pack-from-DST pipelines for the given CB pair.
    init_sfpu(input_cb, output_cb);

    // Outer loop: iterate over blocks (always 1 block for sharded case)
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in output CB for the entire block of tiles upfront.
        // Since the output CB is globally allocated (sharded), this is effectively a no-op
        // but is required for correct CB protocol.
        cb_reserve_back(output_cb, per_core_block_dim);

        // Inner loop: process each tile individually
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to tile register (DST) space.
            // This synchronizes with the packer -- we cannot write to DST
            // while the packer is still reading from it.
            tile_regs_acquire();

            // Wait for 1 tile to be available in the input CB.
            // For sharded data, the reader already pushed all tiles at once,
            // so this will succeed immediately.
            cb_wait_front(input_cb, 1);

            // Unpack one tile from input CB position 0 into DST register 0.
            // The unpacker converts from the CB's data format to the DST
            // register format (FP32 or FP16b depending on fp32_dest_acc_en).
            copy_tile(input_cb, 0, 0);

            // Initialize the SFPU for this specific typecast conversion.
            // This is a macro that expands to typecast_tile_init<IN_FMT, OUT_FMT>().
            // The init function configures SFPLOADMACRO instruction templates,
            // programmable constants, and store format modes.
            TYPECAST_LLK_INIT();

            // Execute the actual typecast on DST register 0.
            // This macro expands to typecast_tile<IN_FMT, OUT_FMT>(0).
            // Depending on the conversion, this may invoke an SFPU kernel
            // or may be a no-op (packer/unpacker-only conversions).
            TYPECAST_LLK(0);

            // Signal that we are done writing to DST registers.
            // Hands control to the packer side of the tile register pipeline.
            tile_regs_commit();

            // Wait for the packer to finish reading from DST.
            tile_regs_wait();

            // Pack the result from DST register 0 into the output CB.
            // The packer converts from DST format to the output CB's data format.
            pack_tile(0, output_cb);

            // Free the consumed tile from the input CB.
            cb_pop_front(input_cb, 1);

            // Release tile register space back to the compute pipeline.
            tile_regs_release();
        }

        // Signal that all tiles in this block have been written to the output CB.
        cb_push_back(output_cb, per_core_block_dim);
    }
}
```

#### Compute Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| `math_fidelity` | `MathFidelity::HiFi4` | Maximum precision for format conversion |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` | Required for 32-bit types (UInt32, Int32, Float32) |
| `unpack_to_dest_mode` | Conditional `UnpackToDestFp32` | When `preserve_fp32_precision` is set, unpacks directly to FP32 in DST |
| `bfp8_pack_precise` | From `args.bfp8_pack_precise` | Enables precise BFP8 packing mode |
| `math_approx_mode` | `false` | Exact conversion, no approximation |

---

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel functions that the compute kernel dispatches to. Because typecast supports many conversion paths, the SFPU implementation is a collection of specialized functions, each handling a specific source-to-destination format pair.

### SFPU Kernel Files

| Architecture | File Path |
|-------------|-----------|
| Wormhole B0 | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h` |
| Blackhole | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h` |
| LLK dispatch (WH) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` |
| LLK dispatch (BH) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` |
| Compute API | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` |

### Architecture of the Dispatch Chain

```
typecast_tile<IN, OUT>(idst)                          [api/compute/eltwise_unary/typecast.h]
  -> llk_math_eltwise_unary_sfpu_typecast<APPROX, IN, OUT>(dst_index)  [llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h]
    -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_typecast_*<APPROX, 8>, ...)
      -> ckernel::sfpu::calculate_typecast_*<APPROX, 8>()              [sfpu/ckernel_sfpu_typecast.h]
        -> _calculate_typecast_*_<APPROX, 8>()                         [tt_llk_*/common/inc/sfpu/ckernel_sfpu_typecast.h]
```

The `if constexpr` chain in `llk_math_eltwise_unary_sfpu_typecast` selects the appropriate `calculate_typecast_*` function at compile time. The template parameter `8` represents `ITERATIONS=8`, meaning 8 rows of 32 elements (= one 32x32 tile processed in row-groups of 4 SFPU lanes).

### Annotated SFPU Kernel Source (Wormhole B0)

The following is the complete Wormhole B0 implementation. The Blackhole variant is structurally identical but uses different ADDR_MOD values (ADDR_MOD_6/ADDR_MOD_7 instead of ADDR_MOD_2/ADDR_MOD_3).

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// ============================================================================
// FP32 -> UINT16
// Throughput: 2 cycles per row via SFPLOADMACRO
// Strategy: Load FP32 from DST, clamp to non-negative via max(v, 0.0),
//           then use stochastic rounding to convert to uint16 with clamp to 65535.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
    // SFPLOADMACRO pipeline schedule (2 cycles per row):
    //   Load -> Simple: max(v, 0.0) -> Round: SFPSTOCHRND FP32_TO_UINT16 -> Store L16
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // Alternate LREG0/LREG1 for pipeline interleaving
        // Load tile row from DST into SFPU local register v, using DEFAULT (FP32) format.
        // SFPLOADMACRO orchestrates the full load->simple->round->store pipeline
        // using macro 0 configured by _init_typecast_fp32_to_uint16_.
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, v >> 2);
        TTI_SFPNOP; // Pipeline bubble: Simple and Round sub-units need time
    }
    TTI_SFPNOP; // Drain pipeline: 3 NOPs to flush remaining in-flight operations
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT16 -> FP16B
// Throughput: 1 cycle per row via SFPLOADMACRO
// Strategy: Load uint16 via LO16 mode, SFPCAST to FP32, then SFPSTOCHRND to FP16B.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
    // SFPLOADMACRO pipeline (1 cycle per row):
    //   Load LO16 -> Simple: cast(v) -> Round: SFPSTOCHRND FP32_TO_FP16B -> Store L16
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1;
        // LO16 load mode: reads lower 16 bits from DST, zero-extending to 32-bit integer
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// INT32 -> FP16B
// Throughput: 4 cycles per row via SFPLOADMACRO
// Strategy: Handle sign separately. Take abs(), extract sign bit to L7,
//           SFPCAST abs value to FP32, restore sign, then SFPMAD to handle
//           the -2^31 edge case, finally SFPSTOCHRND to FP16B.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    constexpr int t = p_sfpu::LREG4; // Temporary register

    // Load constants into LREG0 and LREG1:
    // L0 = 0.0 (used when input is non-negative)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    // L1 = -2^31 (used when input is negative, to correct the SFPCAST result)
    // SFPCAST only handles unsigned integers, so for negative int32:
    //   abs(v) is converted to FP32, then sign is restored, and -2^31 is added
    //   to handle the two's complement wraparound for INT_MIN.
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // Alternate LREG2/LREG3 (LREG0,1 hold constants)
        // Load INT32 row from DST into SFPU register v
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        // t = abs(v): compute absolute value, preserving original in v for sign
        TT_SFPABS(0, v, t, 0);
        // L7 = t >> 31: extract sign bit of the absolute value into LREG7
        // This is used as an indirect index for SFPMAD: L[L7] selects L0 or L1
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5); // SFPSHFT2_MOD1_SHFT_LREG
        // t = cast(t): convert unsigned magnitude to FP32 via SFPCAST
        TTI_SFPCAST(t, t, 0);
        // SFPLOADMACRO then schedules:
        //   setsgn(t, v): restore original sign onto the FP32 magnitude
        //   SFPMAD: L[L7]*1.0 + v (adds 0.0 or -2^31 depending on sign)
        //   SFPSTOCHRND: convert FP32 to FP16B
        //   Store L16
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP; // Extra NOPs for deeper pipeline (4 cycles/row)
}

// ============================================================================
// FP32 -> INT32
// Uses SFPI high-level interface (not SFPLOADMACRO) for clarity.
// Algorithm: extract exponent and mantissa, shift to integer, handle overflow/underflow/sign.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0]; // Load FP32 value from DST row

        // Extract debiased exponent (exp=0 means value in [1,2))
        sfpi::vInt exp = sfpi::exexp(in);

        // Extract mantissa with implicit 1 bit at position 23 (IEEE 754)
        sfpi::vUInt man = sfpi::exman8(in);

        // Shift mantissa to produce integer: shift = (exp - 23)
        // When exp=23, mantissa bits are already in the right position for integer
        sfpi::vInt shift_amt = exp - 23;
        sfpi::vInt result = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(man, shift_amt));

        // Overflow: |value| >= 2^31 -> clamp to INT_MIN (0x80000000)
        v_if (exp >= 31) {
            result = 0x80000000;
        }
        v_endif;

        // Underflow: |value| < 1.0 -> truncate to 0
        v_if (exp < 0) {
            result = 0;
        }
        v_endif;

        // Apply sign: two's complement negation for negative inputs
        v_if (in < 0.0f) {
            result = ~result + 1;
        }
        v_endif;

        sfpi::dst_reg[0] = result; // Store INT32 result back to DST
        sfpi::dst_reg++;           // Advance to next row
    }
}

// ============================================================================
// FP32 -> UINT32
// Uses TTI instructions directly for maximum pipeline efficiency.
// Algorithm: check sign (clamp negatives to 0), extract exp/mantissa, shift.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT, 0); // result = 0

        // LaneEnabled = (in >= 0): negative inputs keep result = 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
        // Extract exponent with condition code: LaneEnabled &= (exp >= 0)
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
        // Overflow: set result to 0xFFFFFFFF
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
        // exp -= 32 (LaneEnabled = exp < 32)
        TTI_SFPIADD(-32 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        // exp += 9: adjust shift for exman8's bit-23 implicit 1
        TTI_SFPIADD(9, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Extract mantissa and shift by computed amount
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0); // Re-enable all lanes

        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

// ============================================================================
// FP32 -> FP16B (round-to-nearest-even with bit-exact rounding)
// Throughput: 3 cycles per row via SFPLOADMACRO
// Strategy: Uses two macros [a] and [b] to implement banker's rounding.
//   Step 1: Right-shift by 16 to get upper 16 bits
//   Step 2: Extract LSB (bit 16 of original = bit 0 after shift)
//   Step 3: Add 0x7FFF (round-half-to-even bias) to copy [b]
//   Step 4: Add LSB + biased copy to produce correctly rounded FP16B
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // Alternate LREG0/LREG1
        // Macro 0 [a]: loads FP32, shifts right by 16, extracts LSB
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_3, a >> 2);
        // Macro 1 [b]: loads same FP32, adds 0x7FFF bias, then adds to [a]
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_2, b >> 2);
        // AND with 1 to isolate the rounding bit
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT16 -> FP32
// Throughput: 1 cycle per row via SFPLOADMACRO
// Strategy: Load uint16 via LO16, SFPCAST to FP32, store as FP32.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp32_()
{
    constexpr int v = p_sfpu::LREG0;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // LO16 load: reads lower 16 bits, SFPCAST converts uint to FP32
        TTI_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// INT32 -> FP32
// Throughput: 4 cycles per row via SFPLOADMACRO
// Strategy: Same as INT32->FP16B but stores FP32 instead of rounding to FP16B.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp32_()
{
    constexpr int t = p_sfpu::LREG4;

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);        // L0 = 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);   // L1 = -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        TT_SFPABS(0, v, t, 0);
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);
        TTI_SFPCAST(t, t, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT32 -> FP16B
// Throughput: 3 cycles per row
// Strategy: Extract sign bit (bit 31 -- always 0 for true uint32 but hardware
//   treats it as sign), clear sign, SFPCAST to FP32, use SFPMAD with L[L7]
//   to add 2^31 back if bit 31 was set, then SFPSTOCHRND to FP16B.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp16b_()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);        // L0 = 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);   // L1 = +2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1);
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        // L7 = v >> 31: extract the MSB (sign bit for SFPCAST purposes)
        TT_SFPSHFT2(v, p_sfpu::LREG12, p_sfpu::LREG7, 5);
        // Clear the sign bit so SFPCAST treats value as positive
        TT_SFPSETSGN(0, v, v, 1);
        // SFPLOADMACRO pipeline then: cast, SFPMAD (add 2^31 if needed), SFPSTOCHRND, store
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT32 -> FP32
// Throughput: 3 cycles per row using 3 SFPLOADMACRO calls
// Strategy: Similar to UINT32->FP16B but final store is FP32.
// Uses three registers [a], [b], [L7] in the macro pipeline.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_fp32_()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00); // 2^31

    constexpr int a  = p_sfpu::LREG2;
    constexpr int b  = p_sfpu::LREG3;
    constexpr int L7 = p_sfpu::LREG7;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // Three macros per iteration:
        // [a]: clear sign bit of loaded INT32
        // [b]: SFPCAST the sign-cleared value to FP32
        // [L7]: shift right by 31 to extract sign, used as SFPMAD indirect index
        TTI_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_3, a >> 2);
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, b >> 2);
        TTI_SFPLOADMACRO((2 << 2) | (L7 & 3), InstrModLoadStore::INT32, ADDR_MOD_2, L7 >> 2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// UINT16 -> UINT32
// Throughput: 1 cycle per row
// Strategy: Load LO16 (zero-extends to 32 bits), store as INT32.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_uint32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        // LO16 load implicitly zero-extends to 32 bits; store as INT32
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::LO16, ADDR_MOD_2, 0);
    }
    TTI_SFPNOP;
}

// ============================================================================
// UINT32 -> UINT16
// Throughput: 2 cycles per row
// Strategy: Load high 16 bits, negate, shift right by 16, OR with low 16 bits.
//   This produces saturated uint16 truncation of the uint32 value.
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint32_to_uint16_()
{
    constexpr int a = p_sfpu::LREG0;
    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 9
    for (int d = 0; d < ITERATIONS + 1; d++)
    {
        int a          = d & 1;
        int macroIndex = 1 + (d & 1);
        if (d < ITERATIONS)
        {
            // Macro 0: load LO16 (high 16 bits due to address offset)
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
            // Macro 1/2: load INT32 from previous row, OR with shifted high bits
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_3, (-4 & 0x3ff) | (b >> 2));
        }
        else
        {
            // Final iteration: use ADDR_MOD_2 for last store
            TTI_SFPLOADMACRO((macroIndex << 2) | (b & 3), InstrModLoadStore::INT32, ADDR_MOD_2, (-2 & 0x3ff) | (b >> 2));
        }
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
}

// ============================================================================
// INT32 -> UINT16
// Throughput: 3 cycles per row
// Strategy: Load INT32, SFPCAST to FP32, clamp negative to 0 via SFPSWAP,
//           then SFPSTOCHRND FP32_TO_UINT16 (clamps to 65535).
// ============================================================================
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_uint16_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1;
        // Load INT32 row from DST
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::INT32, ADDR_MOD_2, a >> 2);
        // Cast INT32 to FP32 for rounding conversion
        TT_SFPCAST(a, a, 0);
        TTI_SFPNOP; // Pipeline delay for cast
        // SFPLOADMACRO pipeline: SFPSWAP max(0, v) -> SFPSTOCHRND -> store L16
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

} // namespace sfpu
} // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Mnemonic | Description |
|------------|----------|-------------|
| `TT_SFPLOADMACRO` / `TTI_SFPLOADMACRO` | SFPLOADMACRO | Macro-scheduled load from DST into SFPU local register, with automatic pipeline scheduling of Simple/MAD/Round/Store sub-units |
| `TTI_SFPLOAD` | SFPLOAD | Direct load from DST into SFPU local register (non-macro) |
| `TTI_SFPSTORE` | SFPSTORE | Direct store from SFPU local register back to DST |
| `TTI_SFPLOADI` | SFPLOADI | Load immediate value into SFPU local register |
| `TTI_SFPNOP` | SFPNOP | Pipeline bubble / synchronization NOP |
| `TTI_SFPCAST` | SFPCAST | Convert unsigned integer to FP32 (INT32_TO_FP32) |
| `TTI_SFP_STOCH_RND` | SFPSTOCHRND | Stochastic/deterministic rounding conversion (FP32_TO_FP16B, FP32_TO_UINT16) |
| `TT_SFPABS` | SFPABS | Compute absolute value of SFPU register |
| `TTI_SFPSHFT2` | SFPSHFT2 | Bit shift (used for sign extraction: `>> 31` to get sign bit into L7) |
| `TTI_SFPSHFT` | SFPSHFT | Variable bit shift (for mantissa alignment in FP32->INT32) |
| `TT_SFPSETSGN` | SFPSETSGN | Set/clear sign bit of a value |
| `TTI_SFPSWAP` | SFPSWAP | Min/max swap; used as `max(0, v)` to clamp negatives |
| `TTI_SFPMAD` | SFPMAD | Multiply-add: `VA * VB + VC`; used with indirect VA for sign-dependent correction |
| `TTI_SFPEXEXP` | SFPEXEXP | Extract debiased exponent from FP32 value |
| `TTI_SFPEXMAN` | SFPEXMAN | Extract mantissa with implicit 1 bit from FP32 value |
| `TTI_SFPSETCC` | SFPSETCC | Set condition codes (lane enable) based on comparison |
| `TTI_SFPENCC` | SFPENCC | Enable all condition codes (re-enable all lanes) |
| `TTI_SFPIADD` | SFPIADD | Integer add (with condition code options and 2's complement support) |
| `TT_SFPAND` | SFPAND | Bitwise AND (used for rounding bit extraction in FP32->FP16B) |
| `TTI_SFPOR` | SFPOR | Bitwise OR (used in UINT32->UINT16 for combining high/low bits) |
| `TTI_SFPGT` | SFPGT | Greater-than comparison (Blackhole: used in UINT32->UINT16 init) |
| `TTI_SFPCONFIG` | SFPCONFIG | Configure SFPLOADMACRO instruction templates and store modes |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | General-purpose / pipeline alternation register A; also holds constant 0.0 for signed conversions |
| `LREG1` | Pipeline alternation register B; also holds constant -2^31 or +2^31 for signed conversions |
| `LREG2` | Pipeline alternation register (for conversions needing L0/L1 as constants); also used as temp `b` |
| `LREG3` | Pipeline alternation register paired with LREG2 |
| `LREG4` | Temporary register `t` for INT32->FP16B and INT32->FP32 (holds abs value during sign processing) |
| `LREG7` | Indirect index register for SFPMAD: holds extracted sign bit (0 or 1) to select L0 or L1 as VA |
| `LREG12` | Programmable constant register (vConstIntPrgm0); holds values like 1, 0x7FFF, or -31 |
| `LREG13` | Programmable constant register (vConstIntPrgm1); holds 0x7FFF for FP32->FP16B rounding |
| `DST registers` | Source/destination tile data; accessed via SFPLOAD/SFPSTORE or `dst_reg[]` in SFPI mode |

### SFPU Execution Flow

For each tile processed by the compute kernel:

1. **Tile register acquire** (`tile_regs_acquire`): The compute kernel acquires exclusive access to DST registers, blocking if the packer is still reading.

2. **CB wait** (`cb_wait_front(input_cb, 1)`): Wait for input tile availability (immediate for sharded data).

3. **Unpack** (`copy_tile(input_cb, 0, 0)`): The unpacker reads one tile from CB c_0 and writes it into DST register 0. The data is converted from the CB's data format to the DST accumulator format (FP32 if `fp32_dest_acc_en`, otherwise FP16b). If `preserve_fp32_precision` is enabled, `UnpackToDestFp32` mode ensures no precision loss during unpack.

4. **SFPU init** (`TYPECAST_LLK_INIT()`): Configures the SFPU for the specific conversion:
   - Programs SFPLOADMACRO instruction templates via `TTI_SFPCONFIG`
   - Sets programmable constants (`vConstIntPrgm0`, `vConstIntPrgm1`)
   - Configures store format mode (FP32, FP16B, INT32, LO16)
   - Programs instruction templates for the Simple, MAD, Round, and Store sub-units

5. **SFPU compute** (`TYPECAST_LLK(0)`): Executes the typecast kernel on DST register 0:
   - Iterates 8 times (ITERATIONS=8), processing 4 rows per iteration (32 rows total = one tile height)
   - Each iteration loads a row from DST, applies the conversion through the SFPU pipeline, and stores back
   - Pipeline depth varies by conversion (1-4 cycles per row), with NOPs for synchronization
   - The `_llk_math_eltwise_unary_sfpu_params_` wrapper handles DST face/row addressing

6. **Tile register commit/wait** (`tile_regs_commit` / `tile_regs_wait`): Hand off DST to the packer and wait for pack completion.

7. **Pack** (`pack_tile(0, output_cb)`): The packer reads from DST register 0 and writes to CB c_2 in the output data format.

8. **CB pop** (`cb_pop_front(input_cb, 1)`): Free the consumed input tile.

9. **Tile register release** (`tile_regs_release`): Release DST back to the compute pipeline.

After all tiles: `cb_push_back(output_cb, per_core_block_dim)` signals the output is complete.

### SFPU Configuration

| Configuration | Value | Purpose |
|---------------|-------|---------|
| `MathFidelity` | HiFi4 | Maximum precision; typecast requires exact conversion |
| `math_approx_mode` | false | No approximation acceptable for format conversion |
| `fp32_dest_acc_en` | Configurable | Must be true when source or target is 32-bit (UInt32, Int32, Float32) |
| `preserve_fp32_precision` | Configurable | Uses `UnpackToDestFp32` unpack mode to avoid FP16b truncation during unpack |
| `bfp8_pack_precise` | Configurable | Enables precise BFP8 packing when target is Bfp8_b |
| `SFPLOADMACRO templates` | Per-conversion | Each init function programs up to 3 macro templates with specific Simple/MAD/Round/Store schedules |
| `vConstIntPrgm0` | Varies | e.g., 1 for FP32->FP16B, -31 for INT32->FP32 |
| `vConstIntPrgm1` | Varies | e.g., 0x7FFF for FP32->FP16B rounding bias |
| `StoreMod0` format | Per-conversion | FP32, FP16B, INT32, or LO16 depending on output format |

### Hardware Compatibility Notes

The Wormhole B0 and Blackhole implementations are **functionally identical** but differ in address modifier usage:

| Feature | Wormhole B0 | Blackhole |
|---------|-------------|-----------|
| Primary ADDR_MOD | `ADDR_MOD_2` | `ADDR_MOD_6` |
| Secondary ADDR_MOD | `ADDR_MOD_3` | `ADDR_MOD_7` |
| `_calculate_typecast_fp32_to_int32_` | Uses SFPI high-level API (`dst_reg[]`, `exexp()`, `shft()`, `v_if`) | Uses TTI instructions (`SFPLOAD`, `SFPEXEXP`, `SFPSHFT`, `SFPSETCC`, `SFPIADD`, `SFPENCC`) |
| `_calculate_typecast_uint32_to_uint16_` | Complex 3-macro approach with `SFPIADD` negate, `SFPSHFT2`, `SFPOR` | Simpler 2-macro approach with `SFPGT` comparison and `SFPOR` |
| SFPGT instruction | Not used | Used in `_init_typecast_uint32_to_uint16_` (`SFPGT_MOD1_SET_VD`) |

The address modifier difference is a hardware register mapping convention between the two architectures. The actual SFPU pipeline behavior is the same.

Notable: The Wormhole `_calculate_typecast_fp32_to_int32_` uses the higher-level SFPI interface (`sfpi::dst_reg`, `sfpi::exexp`, `v_if`), while the Blackhole version uses raw TTI instructions. Both produce equivalent results but the Wormhole version is more readable. This suggests the Wormhole version was refactored to use SFPI after the Blackhole version was written with TTI primitives.

---

## Program Caching and Runtime Override

The `TypecastShardedProgramFactory` supports program caching through the `override_runtime_arguments` method:

```cpp
void TypecastShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams&,
    const TypecastInputs& tensor_args,
    Tensor& output)
```

On cache hit, only the circular buffer addresses are updated via `UpdateDynamicCircularBufferAddress`. The CB handles (`cb_src0`, `out_cb`) are stored in `shared_variables_t`. This avoids re-creating the program, kernels, and CB configurations on repeated invocations with the same shapes/formats.

---

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Typecast operation structure, program factory selection logic, sharded optimization criteria
- `tenstorrent/tt-llk`: SFPU typecast kernel implementations, `ckernel::sfpu` namespace, `calculate_typecast_*` functions, SFPU instruction usage patterns
- `tenstorrent/sfpi`: SFPI programming interface for type conversions, `SFPCAST`, `SFPSTOCHRND` instruction modes, architecture-specific shift operations

### Confluence References
Not consulted for this analysis. The DeepWiki and source code provided sufficient detail on the SFPU instructions used by typecast.

### Glean References
Not consulted for this analysis. The open-source LLK implementations provided sufficient detail on the SFPU kernels.

---

## File Inventory

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.cpp` | Program factory (host-side) |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_sharded_program_factory.hpp` | Program factory header |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.hpp` | Device operation definition |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op_types.hpp` | TypecastParams and TypecastInputs structs |
| `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp` | Compute kernel |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | Reader kernel (shared with unary ops) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h` | Compute API: `typecast_tile<>`, `typecast_tile_init<>` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` | LLK dispatch layer (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h` | LLK dispatch layer (Blackhole) |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` | SFPU wrapper functions (Wormhole) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h` | SFPU wrapper functions (Blackhole) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h` | SFPU kernel implementations (Wormhole) |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h` | SFPU kernel implementations (Blackhole) |
