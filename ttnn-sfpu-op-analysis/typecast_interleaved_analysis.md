# Typecast (Interleaved) Implementation Analysis

## Overview

The **typecast** operation converts tensor elements from one data type to another on Tenstorrent hardware. This analysis covers the `TypecastProgramFactory` — the interleaved (non-sharded) variant used when input/output tensors reside in DRAM with interleaved memory layout. It supports both TILE and ROW_MAJOR tensor layouts.

The operation is unique among SFPU operations because the specific SFPU kernel invoked is selected at compile time based on the source and destination data format pair. Some format conversions (e.g., Bfp8_b to Float16_b) require no SFPU work at all and are handled entirely by the unpacker/packer hardware.

**Program factory path**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp`

## Work Unit Definition

One work unit is **one page**: either one 32x32 tile (for TILE layout) or one row including padding (for ROW_MAJOR layout). The total number of pages (`num_pages`) is obtained from `input.buffer()->num_pages()` and distributed across cores.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|---|---|
| Dimension Convention | N-dimensional (flattened to pages for processing) |
| Tensor Layout | TILE or ROW_MAJOR (detected at runtime via `input.layout()`) |
| Memory Layout | Interleaved |
| Buffer Type | DRAM (interleaved) |
| Data Type | Any supported input dtype (Float16_b, Float32, Int32, UInt16, UInt32, UInt8, Bfp8_b, Bfp4_b) |

### Output Tensor

| Property | Value |
|---|---|
| Dimension Convention | Same shape as input |
| Tensor Layout | Same as input (TILE or ROW_MAJOR) |
| Memory Layout | Interleaved |
| Buffer Type | DRAM (interleaved) |
| Data Type | Target output dtype (Float16_b, Float32, Int32, UInt16, UInt32, UInt8, Bfp8_b, Bfp4_b) |

### Layout Transformations

No tilize/untilize or reshard operations are performed. The operation preserves the tensor layout. The only transformation is the data format conversion applied by the SFPU (and/or unpacker/packer) on each element within each page.

## Data Flow Pattern

1. **Reader** reads one page at a time from DRAM into CB c_0 (input CB) via NoC async read.
2. **Compute** waits for one page in CB c_0, unpacks it into DEST registers via `copy_tile`, runs `TYPECAST_LLK_INIT()` then `TYPECAST_LLK(0)` on the SFPU, packs the result into CB c_2 (output CB), and pops the input page.
3. **Writer** waits for one page in CB c_2, writes it out to DRAM via NoC async write, then pops the output page.

## Circular Buffer Configuration

| CB ID | Name | Data Format | Page Size | Num Pages | Total Size | Buffering | Producer | Consumer |
|---|---|---|---|---|---|---|---|---|
| `c_0` (CB 0) | Input | `cb_data_format_input` (matches input dtype) | `input_page_size` (tile size or row size) | 2 | 2 * input_page_size | Double-buffered | Reader | Compute |
| `c_2` (CB 2) | Output | `cb_data_format_output` (matches output dtype) | `output_page_size` (tile size or row size) | 2 | 2 * output_page_size | Double-buffered | Compute | Writer |

## Pipeline Pattern Summary

Both CBs are double-buffered (capacity = 2 pages, block size = 1 page). This allows the reader to fill one slot while compute processes the other, and similarly for compute/writer. The pipeline permits overlap between all three stages.

## Index Calculations

Index mapping is handled by `TensorAccessor`, which is constructed from compile-time `TensorAccessorArgs` derived from the buffer. The reader iterates pages from `start_id` to `start_id + num_pages`, using `noc_async_read_page(i, s, l1_write_addr)` to translate page index `i` to the correct DRAM bank and address. The writer mirrors this with `noc_async_write_page`.

## Memory Access Patterns

### Read Pattern
Sequential page reads from DRAM. Each core reads a contiguous slice of pages: `[start_id, start_id + num_items_per_core)`. Pages are read one at a time with `noc_async_read_page` followed by `noc_async_read_barrier`.

### Write Pattern
Sequential page writes to DRAM. Each core writes a contiguous slice of pages mirroring the read pattern. Pages are written one at a time with `noc_async_write_page`, flushed per page, with a final `noc_async_write_barrier`.

## Core Distribution Strategy

| Property | Value |
|---|---|
| Grid Topology | Full compute-with-storage grid (`compute_with_storage_grid_size`) |
| Work Splitting | `split_work_to_cores(grid, num_pages, is_row_major)` |
| Core Group 1 | Cores processing `num_items_per_core_group_1` pages each |
| Core Group 2 | Cores processing `num_items_per_core_group_2` pages each (remainder handling) |
| Load Balancing | Two-group strategy: group 1 gets ceil(num_pages/num_cores) pages, group 2 gets floor(num_pages/num_cores) pages |
| Core Ordering | Column-major for TILE layout, row-major for ROW_MAJOR layout (controlled by `is_row_major` flag in `corerange_to_cores`) |

## Arguments

### Compile-Time Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0+ | TensorAccessorArgs | uint32_t[] | Buffer address mode, bank info, page alignment for source tensor |

**Writer kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | output_cb_index | uint32_t | CB index for output (always `c_2` = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Buffer address mode, bank info, page alignment for destination tensor |

**Compute kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (pages) this core processes |
| 1 | per_core_block_dim | uint32_t | Always 1 — one page per block |
| 2 | input_cb | uint32_t | Input CB index (`c_0` = 0) |
| 3 | output_cb | uint32_t | Output CB index (`c_2` = 2) |

### Runtime Arguments

**Reader kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_items_per_core | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | First page index for this core |

**Writer kernel:**

| Index | Name | Type | Description |
|---|---|---|---|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_items_per_core | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | First page index for this core |

## Kernel Implementations

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Shared reader kernel used by many unary operations. Reads pages sequentially from a start ID using TensorAccessor. Gets page size from the CB interface so it works for both tile and row-major layouts. Supports optional `BACKWARDS` define for reverse iteration.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Shared writer kernel. Writes pages one at a time to DRAM. Supports optional `OUT_SHARDED` mode (not used in this interleaved factory). Flushes writes per page for flow control.

### Compute Kernel

This section combines the full annotated source code of the compute kernel with architectural analysis.

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
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // total number of pages (blocks) to process
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // always 1: one page per block
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);            // input circular buffer index (c_0)
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);           // output circular buffer index (c_2)

    init_sfpu(input_cb, output_cb);  // initializes unpack/pack config for the input/output CB pair
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(output_cb, per_core_block_dim);  // reserve 1 output page slot (blocks until writer frees space)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for math

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(input_cb, 1);  // wait for reader to produce 1 input page

            copy_tile(input_cb, 0, 0);  // unpack tile 0 from input_cb into DEST[0] via datacopy LLK

            TYPECAST_LLK_INIT();  // macro-expanded to typecast_tile_init<IN_FMT, OUT_FMT>()
                                  // configures SFPU pipeline for the specific type conversion
            TYPECAST_LLK(0);      // macro-expanded to typecast_tile<IN_FMT, OUT_FMT>(0)
                                  // runs the SFPU typecast kernel on DEST[0]

            tile_regs_commit();  // signal that math is done, DEST registers can be read by packer

            tile_regs_wait();  // wait for packer to be ready to accept tile from DEST

            pack_tile(0, output_cb);  // pack DEST[0] into the output CB in the target data format

            cb_pop_front(input_cb, 1);  // free the consumed input page

            tile_regs_release();  // release DEST registers for next iteration
        }
        cb_push_back(output_cb, per_core_block_dim);  // publish 1 output page for the writer
    }
}
```

### SFPU Kernel Implementation

The typecast operation is unique in that it dispatches to many different SFPU kernel functions depending on the input/output format pair. The dispatch is handled by `llk_math_eltwise_unary_sfpu_typecast<APPROXIMATE, IN_DTYPE, OUT_DTYPE>()` which is a large compile-time if-else chain selecting the appropriate `calculate_typecast_*` function.

#### SFPU Kernel File

**LLK dispatch layer (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`

**Arch-specific wrappers (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`

**Shared SFPU implementations (Wormhole B0)**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`

**Shared SFPU implementations (Blackhole)**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h`

#### Annotated SFPU Kernel Source

Due to the large number of conversion functions, a representative subset is shown below. The full file for the Wormhole B0 shared implementation (`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_typecast.h`) is approximately 936 lines.

**FP32 to UInt16 (SFPLOADMACRO-based, 2 cycles/row):**
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_uint16_()
{
    // Uses SFPLOADMACRO for pipelined execution at 2 cycles per input row.
    // SFPLOADMACRO schedules load -> clamp-to-zero -> round-to-uint16 -> store
    // across the SFPU's Load/Simple/Round/Store sub-units.
    //
    // Pipeline schedule (notation: [x] = scheduled by SFPLOADMACRO with VD=x):
    // t=0: Load [v]
    // t=1: Simple [v] = max(v, 0.0)    -- clamp negatives to zero
    // t=0: (idle)
    // t=1: Round [v] L16 = rnd(v)       -- FP32 to uint16 with clamping at 65535
    // t=0: Store [v] L16

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)  // ITERATIONS=8, processing 8 rows per tile face
    {
        int v = d & 1; // alternate LREG0/LREG1 to allow pipelining
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, v >> 2);
        // SFPLOADMACRO loads from DEST, applies the pre-configured macro (set by init),
        // and stores back to DEST, all with automatic pipeline scheduling.
        TTI_SFPNOP;  // pipeline bubble: must wait for Simple unit to complete
    }
    TTI_SFPNOP;  // drain pipeline: 3 NOPs to flush all in-flight operations
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

**FP32 to Int32 (SFPI high-level, manual bit manipulation):**
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_int32_()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat in = sfpi::dst_reg[0];  // load one row (32 elements) from DEST

        sfpi::vInt exp = sfpi::exexp(in);     // extract debiased exponent (exp=0 means 1.xxx)

        sfpi::vUInt man = sfpi::exman8(in);   // extract mantissa with implicit 1 at bit 23

        sfpi::vInt shift_amt = exp - 23;      // shift amount: positive = left shift, negative = right

        // Shift mantissa to produce integer magnitude
        sfpi::vInt result = sfpi::reinterpret<sfpi::vInt>(sfpi::shft(man, shift_amt));

        // Overflow: if |value| >= 2^31, saturate to INT_MIN
        v_if (exp >= 31) {
            result = 0x80000000;  // INT_MIN
        }
        v_endif;

        // Underflow: if |value| < 1, result is 0
        v_if (exp < 0) {
            result = 0;
        }
        v_endif;

        // Apply sign: two's complement negation for negative inputs
        v_if (in < 0.0f) {
            result = ~result + 1;
        }
        v_endif;

        sfpi::dst_reg[0] = result;  // write back to DEST
        sfpi::dst_reg++;            // advance to next row
    }
}
```

**UInt16 to FP16b (SFPLOADMACRO-based, 1 cycle/row):**
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_uint16_to_fp16b_()
{
    // Fastest conversion: 1 cycle per row using SFPLOADMACRO.
    // Pipeline: Load [v] -> Simple: cast(v) -> Round: rnd(v) -> Store [v] L16

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = d & 1; // alternate registers for pipelining
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::LO16, ADDR_MOD_2, v >> 2);
        // LO16 load mode: reads lower 16 bits of each 32-bit DEST element as uint16
        // Macro applies SFPCAST (int->float) then SFPSTOCHRND (fp32->fp16b rounding)
    }
    TTI_SFPNOP;  // drain pipeline
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

**Int32 to FP16b (SFPLOADMACRO-based, 4 cycles/row):**
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_int32_to_fp16b_()
{
    // Complex conversion requiring sign handling. Uses LREG0=0.0, LREG1=-2^31
    // to correct the sign after unsigned SFPCAST.
    // Algorithm: abs(v) -> shift right by 31 to extract sign bit into L7 ->
    //   cast unsigned to float -> restore sign via setsgn ->
    //   add correction (0.0 or -2^31) via indirect MAD -> round to FP16b

    constexpr int t = p_sfpu::LREG4;  // temporary register

    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_USHORT, 0);       // L0 = 0.0
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0xcf00);  // L1 = -2^31

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int v = 2 + (d & 1); // alternate LREG2/LREG3
        TT_SFPLOADMACRO((0 << 2) | (v & 3), InstrModLoadStore::INT32, ADDR_MOD_2, v >> 2);
        // Load as INT32 from DEST
        TT_SFPABS(0, v, t, 0);
        // t = abs(v), needed because SFPCAST treats input as unsigned
        TTI_SFPSHFT2(t, p_sfpu::LREG12, p_sfpu::LREG7, 5);
        // L7 = t >> 31 (extracts original sign bit, 0 or 1)
        // L7 is used as indirect index: L[L7] selects L0 (0.0) or L1 (-2^31)
        TTI_SFPCAST(t, t, 0);
        // t = cast unsigned int to float
        // Macro then applies: setsgn(t, v) to restore sign,
        //   MAD: L[L7]*1.0 + v to add sign correction,
        //   STOCHRND: FP32 to FP16b rounding,
        //   Store result
    }
    TTI_SFPNOP;  // drain pipeline (5 NOPs for 4-cycle-per-row pipeline depth)
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

**FP32 to FP16b (SFPLOADMACRO-based, 3 cycles/row, round-to-nearest-even):**
```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_typecast_fp32_to_fp16b_()
{
    // Implements IEEE 754 round-to-nearest-even for FP32 -> BFloat16.
    // Uses two SFPLOADMACRO macros (a and b) in an interleaved pipeline:
    //   Macro [a]: shift right 16 -> extract LSB (the "round" bit)
    //   Macro [b]: add 0x7fff bias -> combine with LSB for rounding -> store as BF16

    constexpr int b = p_sfpu::LREG2;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        int a = d & 1; // alternate LREG0/LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_3, a >> 2);
        // Macro 0 [a]: loads from DEST, applies shift-right-16 to get upper 16 bits
        TTI_SFPLOADMACRO((1 << 2) | (b & 3), 0, ADDR_MOD_2, b >> 2);
        // Macro 1 [b]: adds 0x7fff rounding bias, combines with LSB, stores as BF16
        TT_SFPAND(0, p_sfpu::LREG12, a, 0);
        // a &= vConstIntPrgm0 (which is 1): extract the LSB for round-to-nearest-even
    }
    TTI_SFPNOP;  // drain pipeline
    TTI_SFPNOP;
    TTI_SFPNOP;
}
```

#### SFPU Instructions Used

The typecast kernels use a wide range of SFPU instructions across different conversion paths:

| Instruction | Description |
|---|---|
| `SFPLOADMACRO` / `TT_SFPLOADMACRO` | Pipelined load-compute-store macro instruction. Loads from DEST, applies a pre-configured sequence of Simple/MAD/Round/Store operations, and writes back. Enables high-throughput conversions (1-4 cycles per row). |
| `SFPLOAD` / `TTI_SFPLOAD` | Loads a row from DEST register into an SFPU local register (LREG). Supports multiple load modes: DEFAULT (FP32), INT32, LO16 (lower 16 bits). |
| `SFPSTORE` / `TTI_SFPSTORE` | Stores from an SFPU local register back to DEST. Supports INT32 and FP16B store modes. |
| `SFPLOADI` / `TTI_SFPLOADI` | Loads an immediate constant into an SFPU local register. Supports USHORT (16-bit unsigned), SHORT (16-bit signed), FLOATB (BFloat16), LOWER/UPPER (for macro config). |
| `SFPEXEXP` / `TTI_SFPEXEXP` | Extracts the debiased exponent from a floating-point value. Can optionally set condition codes based on exponent sign. |
| `SFPEXMAN` / `TTI_SFPEXMAN` | Extracts the mantissa with implicit leading 1 bit at bit position 23. |
| `SFPIADD` / `TTI_SFPIADD` | Integer add with multiple modes: immediate operand, two's complement negation, condition code control. |
| `SFPSHFT` / `TTI_SFPSHFT` | Barrel shift: shifts an LREG value by the amount in another LREG. Used for mantissa alignment. |
| `SFPSHFT2` / `TTI_SFPSHFT2` | Extended shift instruction with immediate or register shift amounts and special modes (e.g., shift LREG). |
| `SFPCAST` / `TTI_SFPCAST` | Converts unsigned integer to IEEE 754 float (uint -> fp32). |
| `SFPSETSGN` / `TT_SFPSETSGN` | Sets the sign bit of a floating-point value from another register or immediate. |
| `SFPABS` / `TT_SFPABS` | Computes absolute value (clears sign bit). |
| `SFPMAD` / `TTI_SFPMAD` | Multiply-accumulate: VA * VB + VC. Supports indirect VA selection via L7 register. |
| `SFPSETCC` / `TTI_SFPSETCC` | Sets lane-enable condition codes based on register value (LT0, GTE0). |
| `SFPENCC` / `TTI_SFPENCC` | Enables all lanes (clears condition codes). |
| `SFP_STOCH_RND` / `TTI_SFP_STOCH_RND` | Stochastic/deterministic rounding. Modes include FP32_TO_FP16B and FP32_TO_UINT16. |
| `SFPSWAP` / `TTI_SFPSWAP` | Min/max swap operation. Used to clamp negative values to zero. |
| `SFPAND` / `TTI_SFPAND` | Bitwise AND between registers. Used for masking (e.g., extracting LSB, masking to 8 bits). |
| `SFPOR` / `TTI_SFPOR` | Bitwise OR between registers. Used in uint32-to-uint16 conversion. |
| `SFPGT` / `TTI_SFPGT` | Greater-than comparison. Used in Blackhole uint32-to-uint16 init. |
| `SFPNOP` / `TTI_SFPNOP` | No-operation. Required for pipeline draining and hazard avoidance. |
| `SFPCONFIG` / `TTI_SFPCONFIG` | Configures SFPLOADMACRO: defines macro instruction sequences, store formats, and delay settings. |

#### SFPU Register Usage

| Register | Usage |
|---|---|
| `LREG0` - `LREG3` | Primary working registers. Most conversions alternate between LREG0/LREG1 (or LREG2/LREG3 for int32 conversions) to enable pipeline overlap. |
| `LREG4` (`t`) | Temporary register for int32 conversions — holds abs(value) during sign extraction. |
| `LREG7` | Indirect index register for SFPMAD. Stores the sign bit (0 or 1) extracted via SFPSHFT2. Used by `SFPMAD_MOD1_INDIRECT_VA` to select between L0 and L1 as the addend. |
| `LREG12` (`vConstIntPrgm0`) | Programmable constant register. Set to different values per conversion: 1 (for FP32->FP16b LSB extraction), 0xFF (for uint8 masking), -31 (for shift amounts). |
| `LREG13` (`vConstIntPrgm1`) | Programmable constant register. Set to 0x7FFF for FP32->FP16b rounding bias. |
| `DEST registers` | The source and destination of data. Tiles are unpacked into DEST by `copy_tile`, modified by SFPU operations via `SFPLOAD`/`SFPSTORE` or `SFPLOADMACRO`, and then packed out by `pack_tile`. |

#### SFPU Execution Flow

1. **Tile acquisition**: `cb_wait_front(input_cb, 1)` blocks until the reader has produced one page in the input CB.
2. **Unpack to DEST**: `copy_tile(input_cb, 0, 0)` calls `llk_unpack_A` to unpack page 0 from CB into DEST[0], then `llk_math_eltwise_unary_datacopy` to move data within DEST.
3. **SFPU initialization**: `TYPECAST_LLK_INIT()` calls `typecast_tile_init<IN_FMT, OUT_FMT>()` which dispatches to the appropriate `llk_math_eltwise_unary_sfpu_typecast_init`. This configures:
   - SFPU address modifiers (ADDR_MOD_2/3/6/7 for auto-incrementing DEST row pointers)
   - SFPLOADMACRO instruction templates and macros via `SFPCONFIG`/`SFPLOADI`
   - Programmable constants (`vConstIntPrgm0`, `vConstIntPrgm1`)
   - SFPU misc configuration (store format, delay kind)
4. **SFPU execution**: `TYPECAST_LLK(0)` calls `typecast_tile<IN_FMT, OUT_FMT>(0)` which dispatches to the appropriate `calculate_typecast_*` function. This iterates over 8 rows (ITERATIONS=8) per tile face, processing all 4 faces of a 32x32 tile. The `_llk_math_eltwise_unary_sfpu_params_` wrapper handles face iteration and DEST pointer management.
5. **Pack to output**: `pack_tile(0, output_cb)` packs DEST[0] into the output CB using the output data format.
6. **Page release**: `cb_pop_front(input_cb, 1)` frees the input page; `cb_push_back(output_cb, 1)` publishes the output page.

#### SFPU Configuration

| Configuration | Value |
|---|---|
| Math Fidelity | `HiFi4` (highest fidelity, no approximation) |
| Math Approx Mode | `false` (exact mode) |
| FP32 DEST Accumulation | Controlled by `args.fp32_dest_acc_en` — enabled when either input or output is 32-bit |
| Unpack-to-DEST Mode | `UnpackToDestFp32` when `preserve_fp32_precision` is true, else `Default` |
| BFP8 Pack Precise | Controlled by `args.bfp8_pack_precise` — enables precise BFP8 packing |

The `TYPECAST_LLK_INIT` and `TYPECAST_LLK` macros are defined via C++ preprocessor defines in the program factory:
```cpp
unary_defines["TYPECAST_LLK_INIT"] = "typecast_tile_init<IN_FMT, OUT_FMT>";
unary_defines["TYPECAST_LLK"] = "typecast_tile<IN_FMT, OUT_FMT>";
```
where `IN_FMT` and `OUT_FMT` are the numeric values of `tt::DataFormat` for the input and output types.

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The SFPLOADMACRO-based implementations are structurally identical between both architectures, with the only differences being in ADDR_MOD indices (Wormhole uses ADDR_MOD_2/3, Blackhole uses ADDR_MOD_6/7). This reflects different address modifier configurations in the two architectures.
- **Blackhole-specific**: The `_calculate_typecast_uint32_to_uint16_` and `_init_typecast_uint32_to_uint16_` use `SFPGT` instruction on Blackhole versus `SFPIADD` with two's complement on Wormhole, reflecting instruction set differences.
- **FP32 to Int32**: The Wormhole implementation uses the SFPI high-level API (`sfpi::dst_reg`, `sfpi::exexp`, `sfpi::exman8`, `sfpi::shft`, `v_if`/`v_endif`), while the Blackhole implementation uses low-level TTI instructions. The Wormhole version has a commented-out TTI version as well.
- **No-SFPU conversions**: Several format pairs require no SFPU kernel at all and are handled purely by the unpacker/packer hardware: Bfp8_b <-> Float16_b, Bfp8_b <-> Float32, Bfp4_b <-> Float16_b, Bfp4_b <-> Bfp8_b, Bfp4_b <-> Float32, Float16_b -> Float32, Float32 -> Bfp8_b, Float16_b -> Bfp8_b, UInt8 -> Int32/UInt32. In these cases, `copy_tile` moves data to DEST and `pack_tile` converts the format during packing.

## Implementation Notes

1. **Dual layout support**: The program factory handles both TILE and ROW_MAJOR layouts. For ROW_MAJOR, the page size comes from the buffer rather than being computed from tile size. The `per_core_block_dim` is always 1 regardless of layout, meaning each "block" is exactly one page.

2. **Program caching**: The factory implements `override_runtime_arguments` to efficiently update buffer addresses across calls without recreating the full program. Only `src_buffer->address()` and `dst_buffer->address()` are updated.

3. **SFPLOADMACRO optimization**: Many conversion paths use the SFPLOADMACRO instruction, which achieves high throughput (1-4 cycles per row) by scheduling multiple SFPU sub-units (Load, Simple, MAD, Round, Store) in parallel. This is significantly faster than manually issuing individual SFPU instructions.

4. **Register alternation**: Most SFPLOADMACRO-based kernels alternate between two local registers (e.g., LREG0/LREG1) across iterations, allowing the pipeline to overlap processing of one row while loading the next.

5. **Indirect MAD for sign handling**: The int32/uint32 to float conversions use an indirect SFPMAD via L7 as an index register. The sign bit (0 or 1) stored in L7 selects between L0 (0.0 or correction constant) and L1 (-2^31 or 2^31) as the addend, implementing sign correction without branching.

6. **SubgridProgramFactory**: The file also contains `TypecastSubgridProgramFactory` which operates on a user-specified subset of cores (`sub_core_grids`). It requires uniform tile distribution (ntiles must be divisible by ncores) and uses block-level processing rather than single-page processing.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the typecast operation work in TTNN? What kernels does it use and how is the program factory structured for interleaved typecast?"
   **Reason**: Initial reconnaissance to understand the operation's architecture, program factory selection logic, and kernel structure.
   **Key Findings**: Identified four program factories (TypecastProgramFactory, TypecastSubgridProgramFactory, TypecastRowMajorChunkedProgramFactory, TypecastShardedProgramFactory). Confirmed that SFPU kernels are selected based on input/output format pairs, and some conversions are handled by unpacker/packer without SFPU.

2. **Query**: "How does typecast_tile and typecast_tile_init work in the LLK? What SFPU functions do they call? Where is llk_math_eltwise_unary_sfpu_typecast defined?"
   **Reason**: Needed to trace the dispatch chain from the compute kernel API down to the actual SFPU implementation functions.
   **Key Findings**: Confirmed the dispatch pattern: `typecast_tile<IN,OUT>(idx)` -> `llk_math_eltwise_unary_sfpu_typecast<APPROX, IN, OUT>(idx)` -> `_llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_typecast_*, idx, mode)`. Identified `ckernel_sfpu_typecast.h` as the file containing the actual SFPU implementations, and learned about the SFPLOADMACRO pattern used for high-throughput conversions.

### Documentation References

1. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`
   **Reason**: To understand the public compute API for typecast
   **Key Information**: `typecast_tile<IN_DTYPE, OUT_DTYPE>(idst)` wraps `llk_math_eltwise_unary_sfpu_typecast`, and `typecast_tile_init` wraps `llk_math_eltwise_unary_sfpu_typecast_init`. Both are template-parameterized on IN_DTYPE and OUT_DTYPE as uint32_t (DataFormat enum values).

2. **Source**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op_types.hpp`
   **Reason**: To understand the TypecastParams structure and available configuration options
   **Key Information**: Identified `fp32_dest_acc_en`, `preserve_fp32_precision`, `bfp8_pack_precise`, and `sub_core_grids` as key parameters controlling typecast behavior.
