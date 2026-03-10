# TYPECAST (Row-Major Chunked) Implementation Analysis

## Overview

The typecast row-major chunked operation converts tensor data from one data type to another while maintaining row-major layout. Unlike the standard interleaved typecast (which works on tiles), this variant operates on row-major data by splitting each row into fixed-size "chunks" of up to 1024 elements. Each chunk is read from DRAM, passed through the SFPU for type conversion, and written back to DRAM. The operation supports a wide range of type conversions including Float16_b, Float32, Int32, UInt16, UInt32, UInt8, Bfp8_b, and Bfp4_b.

**Program factory path**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_rm_chunked_program_factory.cpp`

This factory is selected when the input tensor has `Layout::ROW_MAJOR`. It is enforced with a `TT_FATAL` assertion at the top of the `create` method.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | chunk (sub-row segment) |
| **Unit size** | Up to 1024 elements (capped by `max_elements_per_chunk`) |
| **Total units** | `num_rows * (full_chunks_per_row + partial_chunks_per_row)` |
| **Loop structure** | Outer loop over rows, inner loop over chunks per row (full then partial) |

One work unit is a single chunk of a row. Rows are split into full chunks of `min(1024, row_width_elements)` elements, plus an optional partial chunk for the remainder. The compute kernel sees each chunk as a single "block" (one page in/out of circular buffers).

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Arbitrary (any rank); last dimension = row width |
| **Dimension convention** | Generic (row = page, last dim = contiguous elements) |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1, via TensorAccessor) |
| **Data type** | Any supported source dtype (Float16_b, Float32, Int32, UInt16, UInt32, UInt8, Bfp8_b, Bfp4_b) |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | ROW_MAJOR |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1, via TensorAccessor) |
| **Data type** | Target dtype (Float16_b, Float32, Int32, UInt16, UInt32, UInt8, Bfp8_b, Bfp4_b) |

### Layout Transformations

No tilize/untilize or reshard operations are performed. Data remains in row-major layout throughout. The type conversion happens in the SFPU via the unpack-compute-pack pipeline: the unpacker interprets raw bytes in the source format, the SFPU performs any necessary bit manipulation for the conversion, and the packer writes bytes in the destination format.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM (src_buffer via TensorAccessor) | CB c_0 (input) | `cb_reserve_back`, `noc_async_read`, `noc_async_read_barrier`, `cb_push_back` |
| 2 | Compute | CB c_0 (input) | CB c_2 (output) | `cb_wait_front`, `copy_tile` (unpack to DEST), SFPU typecast, `pack_tile`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back` |
| 3 | Writer | CB c_2 (output) | DRAM (dst_buffer via TensorAccessor) | `cb_wait_front`, `noc_async_write`, `noc_async_writes_flushed`, `cb_pop_front` |

**Step-by-step flow**:
1. The reader kernel iterates over its assigned rows. For each row, it reads full chunks (each up to 1024 elements) from DRAM into CB c_0 one at a time, using `TensorAccessor::get_noc_addr` with a byte offset to address sub-row regions. If the row has a remainder, a partial chunk is read as well.
2. The compute kernel loops over the total number of chunks assigned to the core. For each chunk it: acquires tile registers, waits for a chunk in CB c_0, copies it to DEST registers via `copy_tile`, runs the SFPU typecast operation, commits/waits on tile registers, packs the result to CB c_2, and pops the input.
3. The writer kernel mirrors the reader: it iterates over rows and chunks, waiting for data in CB c_2, writing each chunk to the appropriate DRAM address with byte offset, and popping the CB.
4. The writer issues a final `noc_async_write_barrier` to ensure all writes are flushed.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input staging (raw source-format chunk) | 2 pages x `input_full_chunk_size_bytes` | 1 page (`input_full_chunk_size_bytes`) | Double | Reader | Compute | Program |
| c_2 | cb_output | Output staging (converted dest-format chunk) | 2 pages x `output_full_chunk_size_bytes` | 1 page (`output_full_chunk_size_bytes`) | Double | Compute | Writer | Program |

**Notes**:
- Page size is set to `full_chunk_size_bytes` (the maximum chunk size). For partial chunks, the reader/writer transfer fewer bytes but the CB page allocation remains at the full chunk size.
- Both CBs use double buffering (capacity = 2 pages), allowing overlap between reader/compute and compute/writer.
- The data format of c_0 matches the input dtype; the data format of c_2 matches the output dtype.

## Pipeline Pattern Summary

Both circular buffers are double-buffered (capacity = 2x block size). This allows the reader to fill the next chunk while the compute kernel processes the current one, and similarly the compute kernel can produce the next output while the writer drains the current one. This creates a 3-stage pipelined execution: Read(N+1) | Compute(N) | Write(N-1).

## Index Calculations

The program factory uses `TensorAccessor` for address computation in both reader and writer kernels. The key index mapping is:

- **Row index** (`row_id`): Maps to a page in the interleaved buffer. `TensorAccessor::get_noc_addr(row_id, byte_offset)` returns the NoC address for page `row_id` at the given byte offset within that page.
- **Chunk offset**: `chunk_idx * full_chunk_size_bytes` for full chunks, or `full_chunks_per_row * full_chunk_size_bytes` for the partial chunk. This byte offset is passed as the second argument to `get_noc_addr`.
- **Row page size** (`row_page_size_bytes`): Passed to TensorAccessor to describe the buffer's actual page layout, which equals `row_width_elements * element_size_bytes`.

The mapping is: `logical_position = (row_id, element_offset_within_row)` -> `physical_address = bank_base + row_id * row_page_size + chunk_byte_offset`.

## Memory Access Patterns

### Read Pattern

Sequential within each row: chunks are read left-to-right (increasing byte offset). Rows are processed sequentially from `start_row_id` to `end_row_id`. Each NoC read transfers exactly `full_chunk_size_bytes` (or `partial_chunk_size_bytes` for the remainder). A `noc_async_read_barrier` is issued after each individual chunk read, meaning reads are not overlapped across chunks (they are serialized per chunk).

### Write Pattern

Mirrors the read pattern: sequential chunk-by-chunk within each row, rows processed in order. Each chunk write is followed by `noc_async_writes_flushed` (not a full barrier), and a single `noc_async_write_barrier` at the very end ensures all writes complete before the kernel exits.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (row-wise) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | `min(num_rows, max_cores)` |
| **Work per core** | `num_rows_per_core_group_1` or `num_rows_per_core_group_2` rows |
| **Load balancing** | Two-group: group 1 gets `ceil(num_rows / num_cores)` rows, group 2 gets `floor(num_rows / num_cores)` rows |

Work is distributed at **row granularity** using `split_work_to_cores`. Each core handles complete rows (all chunks within those rows). The `row_wise=true` parameter means cores are filled row-first across the grid. If rows do not divide evenly, `core_group_1` gets one extra row per core compared to `core_group_2`.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_in | uint32_t | Input circular buffer index (c_0) |
| 1 | full_chunk_size_bytes | uint32_t | Size of a full chunk in bytes (input format) |
| 2 | full_chunks_per_row | uint32_t | Number of full chunks per row |
| 3 | partial_chunk_size_bytes | uint32_t | Size of the partial (remainder) chunk in bytes |
| 4 | partial_chunks_per_row | uint32_t | 0 or 1 -- whether a partial chunk exists |
| 5 | row_page_size_bytes | uint32_t | Full row size in bytes (buffer page size) |
| 6+ | TensorAccessorArgs | varies | Appended by `TensorAccessorArgs(*src_buffer)` |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (c_2) |
| 1 | full_chunk_size_bytes | uint32_t | Size of a full chunk in bytes (output format) |
| 2 | full_chunks_per_row | uint32_t | Number of full chunks per row |
| 3 | partial_chunk_size_bytes | uint32_t | Size of the partial (remainder) chunk in bytes |
| 4 | partial_chunks_per_row | uint32_t | 0 or 1 -- whether a partial chunk exists |
| 5 | row_page_size_bytes | uint32_t | Full row size in bytes (buffer page size) |
| 6+ | TensorAccessorArgs | varies | Appended by `TensorAccessorArgs(*dst_buffer)` |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Total chunks to process on this core = `num_rows * chunks_per_row_total` |
| 1 | per_core_block_dim | uint32_t | Always 1 (one chunk per block iteration) |
| 2 | input_cb | uint32_t | Input circular buffer index (c_0) |
| 3 | output_cb | uint32_t | Output circular buffer index (c_2) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM |
| 1 | num_rows | uint32_t | Number of rows assigned to this core |
| 2 | start_row_id | uint32_t | Starting row index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM |
| 1 | num_rows | uint32_t | Number of rows assigned to this core |
| 2 | start_row_id | uint32_t | Starting row index for this core |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM (src_buffer) | CB c_0 | Read row chunks via TensorAccessor |
| compute | RISCV_2 (MATH) | N/A | CB c_0 | CB c_2 | Unpack, SFPU typecast, pack |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM (dst_buffer) | Write row chunks via TensorAccessor |

### Reader Kernel

- **File**: `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp`
- **Key Logic**: Iterates over assigned rows (`start_row_id` to `end_row_id`). For each row, processes all full chunks sequentially, then the partial chunk (if it exists, via `if constexpr`). Uses `TensorAccessor::get_noc_addr(row_id, byte_offset)` for sub-page addressing. Each chunk read is immediately barrier'd and pushed.

### Writer Kernel

- **File**: `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp`
- **Key Logic**: Mirrors the reader structure exactly. Uses `noc_async_writes_flushed()` (lightweight flush) per chunk and a single `noc_async_write_barrier()` at kernel exit. The partial chunk branch uses `if constexpr` for zero-overhead elimination when there is no remainder.

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
    // per_core_block_cnt: total number of chunks this core must process
    // (num_rows_for_core * chunks_per_row_total)
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);

    // per_core_block_dim: always 1 -- each "block" is a single chunk
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    // input_cb: CB index for input data (c_0)
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);

    // output_cb: CB index for output data (c_2)
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);

    // Initializes the unpack-math-pack pipeline for SFPU unary operations.
    // Calls unary_op_init_common which configures unpacker for input_cb,
    // packer for output_cb, and synchronization primitives.
    init_sfpu(input_cb, output_cb);

    // Main loop: iterate over all chunks assigned to this core
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in output CB for per_core_block_dim pages (always 1).
        // This blocks until the writer has consumed enough output pages.
        cb_reserve_back(output_cb, per_core_block_dim);

        // Inner loop over tiles within a block (always 1 iteration here)
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to DEST registers for the math RISC.
            // Prevents the packer from reading DEST while math is writing.
            tile_regs_acquire();

            // Wait until the reader has pushed 1 page (chunk) into input CB.
            cb_wait_front(input_cb, 1);

            // Unpack the chunk from input CB into DEST register 0.
            // copy_tile performs A2D (source A to DEST) datacopy via the unpacker.
            // The unpacker interprets raw bytes according to the input CB's data format.
            copy_tile(input_cb, 0, 0);

            // Initialize the SFPU for the specific typecast conversion.
            // TYPECAST_LLK_INIT is a preprocessor define set by the program factory to
            // typecast_tile_init<IN_DTYPE, OUT_DTYPE>(), which configures SFPU
            // instruction templates (SFPLOADMACRO, SFPSWAP, SFPCAST, SFP_STOCH_RND, etc.)
            // for the particular type conversion direction.
            TYPECAST_LLK_INIT();

            // Execute the SFPU typecast on tile in DEST[0].
            // TYPECAST_LLK is a preprocessor define set to typecast_tile<IN_DTYPE, OUT_DTYPE>(0).
            // This dispatches to the appropriate _calculate_typecast_*_ function via
            // _llk_math_eltwise_unary_sfpu_params_, which iterates over tile faces (8 iterations)
            // executing SFPU instructions to convert data in-place in DEST registers.
            // For some conversions (e.g., Float16_b <-> Bfp8_b), no SFPU work is needed
            // because the unpack/pack stages handle the format change.
            TYPECAST_LLK(0);

            // Signal that math is done writing DEST; packer can now read.
            tile_regs_commit();

            // Wait for packer to become available (previous pack completed).
            tile_regs_wait();

            // Pack DEST[0] into the output CB, interpreting the data according
            // to the output CB's data format.
            pack_tile(0, output_cb);

            // Pop the consumed input chunk from input CB, freeing space for reader.
            cb_pop_front(input_cb, 1);

            // Release DEST registers so next iteration's tile_regs_acquire can proceed.
            tile_regs_release();
        }

        // Push the completed output block (1 page) to output CB for writer to consume.
        cb_push_back(output_cb, per_core_block_dim);
    }
}
```

### SFPU Kernel Implementation

The typecast operation is unique among SFPU operations because it does not have a single SFPU kernel function. Instead, it dispatches to one of many conversion-specific SFPU kernel functions based on the input/output data type pair. The dispatch is fully resolved at compile time through the `TYPECAST_LLK` and `TYPECAST_LLK_INIT` preprocessor defines.

#### SFPU Kernel File

**Dispatch layer**: `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`
**LLK routing layer**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`
**SFPU implementation (arch-specific wrappers)**: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`
**SFPU implementation (shared core logic)**: `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_typecast.h` (in tt-llk submodule, not checked out locally)

#### Annotated SFPU Kernel Source

The arch-specific wrapper file (Wormhole B0 shown; Blackhole is nearly identical) provides thin wrappers that delegate to shared `_calculate_typecast_*_` and `_init_typecast_*_` functions from the tt-llk submodule. Two functions are implemented directly in the wrapper for UInt8 conversions.

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h
// (Blackhole version is at the same relative path under blackhole/)

// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_typecast.h"  // from tt-llk submodule: _calculate_typecast_*_ functions

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// --- Wrappers delegating to tt-llk shared implementations ---

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint16() {
    // Delegates to tt-llk: uses TT_SFPLOADMACRO + TTI_SFPNOP loop
    // Loads FP32 values from DEST, SFPSWAP clamps to >=0, SFP_STOCH_RND rounds to UINT16
    _calculate_typecast_fp32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

// --- FP32 to UINT8: directly implemented (not in tt-llk) ---
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        // Load FP32 value from DEST into LREG0
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);
        // Extract biased exponent from FP32 into LREG2
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        // Extract 8-bit mantissa (with implicit 1) from FP32 into LREG1
        TTI_SFPEXMAN(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        // Subtract 23 from exponent to get the shift amount for integer conversion
        // (FP32 has 23 mantissa bits; shift = exponent - bias - 23 gives integer part)
        TTI_SFPIADD(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2,
                     sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Shift mantissa right by (23 - exponent) to produce the integer magnitude
        TTI_SFPSHFT(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Set condition code: lane enabled if input < 0
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
        // For negative values: negate via two's complement (~mantissa + 1)
        TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG1,
                     sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        // Add 256 to wrap negative results into 0-255 range (unsigned saturation)
        TTI_SFPIADD(256, p_sfpu::LREG1, p_sfpu::LREG1,
                     sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Re-enable all lanes
        TTI_SFPENCC(0, 0, 0, 0);
        // Mask to 8 bits: AND with 0xFF (stored in LREG12 = vConstIntPrgm0)
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        // Store result as INT32 back to DEST
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

// --- UINT/INT to UINT8: directly implemented ---
template <bool APPROXIMATION_MODE, int ITERATIONS, bool u16 = false>
inline void calculate_typecast_uint_to_uint8() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d) {
        if constexpr (u16) {
            // Load as 16-bit unsigned (zero-extended to 32-bit in LREG)
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::LO16, ADDR_MOD_3, 0);
        } else {
            // Load as 32-bit integer
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        }
        // Add 256 to ensure proper wrapping for values > 255
        TTI_SFPIADD(256, p_sfpu::LREG0, p_sfpu::LREG0,
                     sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        // Mask to low 8 bits: AND with 0xFF (LREG12 = vConstIntPrgm0)
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG0, 0);
        // Store result as INT32
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, 0);
    }
}

// --- Remaining conversion wrappers (all delegate to tt-llk) ---
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp16b() {
    _calculate_typecast_uint16_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp16b() {
    _calculate_typecast_int32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_int32() {
    _calculate_typecast_fp32_to_int32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_fp16b() {
    _calculate_typecast_fp32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_fp32() {
    _calculate_typecast_uint16_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_fp32() {
    _calculate_typecast_int32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_fp32_to_uint32() {
    _calculate_typecast_fp32_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp16b() {
    _calculate_typecast_uint32_to_fp16b_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_fp32() {
    _calculate_typecast_uint32_to_fp32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint16_to_uint32() {
    _calculate_typecast_uint16_to_uint32_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_uint32_to_uint16() {
    _calculate_typecast_uint32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_typecast_int32_to_uint16() {
    _calculate_typecast_int32_to_uint16_<APPROXIMATION_MODE, ITERATIONS>();
}

// --- Init functions ---

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_fp16b() {
    _init_typecast_fp32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_uint32() {
    _init_typecast_uint16_to_uint32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp32() {
    _init_typecast_uint32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp32() {
    _init_typecast_int32_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp32() {
    _init_typecast_uint16_to_fp32_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint16_to_fp16b() {
    _init_typecast_uint16_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_fp16b() {
    _init_typecast_int32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_fp16b() {
    _init_typecast_uint32_to_fp16b_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint16() {
    _init_typecast_fp32_to_uint16_<APPROXIMATION_MODE>();
}

// UInt8 init functions: directly set the programmable constant register to 0xFF mask
template <bool APPROXIMATION_MODE>
inline void init_typecast_fp32_to_uint8() {
    // Store 0xFF bitmask in SFPU programmable constant register (LREG12)
    // Used by SFPAND instruction to mask results to 8 bits
    sfpi::vConstIntPrgm0 = 0xFF;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint_to_uint8() {
    // Same 0xFF bitmask for integer-to-uint8 paths
    sfpi::vConstIntPrgm0 = 0xFF;
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_uint32_to_uint16() {
    _init_typecast_uint32_to_uint16_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void init_typecast_int32_to_uint16() {
    _init_typecast_int32_to_uint16_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
```

#### SFPU Instructions Used

The typecast operation employs a wide variety of SFPU instructions depending on the conversion path. The key instructions observed across all conversion functions:

| Instruction | Description |
|-------------|-------------|
| `TTI_SFPLOAD` / `TT_SFPLOADMACRO` | Load data from DEST registers into SFPU local registers (LREG0-LREG3). Load mode varies: `DEFAULT` for FP32, `INT32` for integers, `LO16` for 16-bit values. |
| `TTI_SFPSTORE` | Store SFPU local register contents back to DEST registers. Store mode (`INT32`, `DEFAULT`) determines output interpretation. |
| `TTI_SFPEXEXP` | Extract biased exponent from an FP32 value in an LREG. |
| `TTI_SFPEXMAN` | Extract mantissa (with implicit leading 1) from an FP32 value. |
| `TTI_SFPIADD` | Integer add with immediate or register operand. Used for exponent adjustment, two's complement negation, and constant addition. |
| `TTI_SFPSHFT` | Arithmetic/logical shift by amount in another register. Used to convert mantissa to integer. |
| `TTI_SFPSETCC` | Set condition codes based on register value (e.g., less than zero). Enables per-lane conditional execution. |
| `TTI_SFPENCC` | Enable all lanes (clear condition code masking). |
| `TTI_SFPAND` | Bitwise AND between two registers. Used for masking (e.g., `& 0xFF`). |
| `TTI_SFPSWAP` | Swap/min/max operation. Used in init to clamp values to non-negative for unsigned conversions. |
| `TTI_SFP_STOCH_RND` | Stochastic rounding instruction. Used for FP32-to-UInt16, FP32-to-FP16b, and similar precision-reducing conversions. |
| `TTI_SFPCAST` | Type cast instruction for INT32<->FP32 conversions. |
| `TTI_SFPSHFT2` | Secondary shift instruction used for sign bit extraction in INT32->FP32 paths. |
| `TTI_SFPABS` | Absolute value operation, used in INT32->FP32 conversion. |
| `TTI_SFPLOADI` | Load immediate constant into SFPU register. |
| `TTI_SFPCONFIG` | Configure SFPU macro modes and store modes. |
| `TTI_SFPMAD` | Multiply-add operation, used in some conversion paths for scaling. |
| `TTI_SFPNOP` | No-operation, used for pipeline timing/delay after SFPLOADMACRO. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` (p_sfpu::LREG0) | Primary data register: loaded from DEST, holds input value |
| `LREG1` (p_sfpu::LREG1) | Secondary data register: holds mantissa or intermediate results |
| `LREG2` (p_sfpu::LREG2) | Tertiary register: holds exponent or shift amounts |
| `LREG3` (p_sfpu::LREG3) | Auxiliary register: used in some conversions for sign handling |
| `LREG12` (p_sfpu::LREG12) | Programmable constant register (`vConstIntPrgm0`): holds 0xFF mask for UInt8 paths |
| `LCONST_0` | Hardware constant register containing 0, used for two's complement negation |
| `DEST[0]` | Destination register tile: holds the data being converted. Unpacker writes here, packer reads from here. |

#### SFPU Execution Flow

1. **Acquire DEST**: `tile_regs_acquire()` gives the math RISC exclusive access to DEST registers.
2. **Unpack to DEST**: `copy_tile(input_cb, 0, 0)` triggers the unpacker to read from CB c_0 and write the data into `DEST[0]`, converting from the wire format (input dtype) to the DEST register format (FP32 or FP32-accumulation mode).
3. **Init SFPU**: `TYPECAST_LLK_INIT()` configures SFPU instruction templates and macro registers for the specific conversion. Some conversions configure `SFPLOADMACRO` templates, stochastic rounding modes, or load constants.
4. **Execute SFPU**: `TYPECAST_LLK(0)` dispatches to the appropriate `calculate_typecast_*` function. The function loops 8 times (ITERATIONS=8), processing 8 rows of the tile face per iteration (32x32 tile = 4 faces x 8 rows x 32 columns, but SFPU processes 32-wide vectors, so 8 iterations covers one face). The `_llk_math_eltwise_unary_sfpu_params_` wrapper handles iterating across all 4 faces of the tile.
5. **Commit DEST**: `tile_regs_commit()` signals math is done.
6. **Wait for packer**: `tile_regs_wait()` blocks until the packer is ready.
7. **Pack from DEST**: `pack_tile(0, output_cb)` reads from `DEST[0]` and writes to CB c_2, converting to the output wire format.
8. **Release**: `cb_pop_front` frees the input CB page, `tile_regs_release()` frees DEST for the next iteration.

#### SFPU Configuration

| Configuration | Value | Notes |
|---------------|-------|-------|
| **Math fidelity** | HiFi4 | Highest fidelity; relevant for unpacker/packer precision |
| **Math approx mode** | false | No approximation (exact conversions required) |
| **fp32_dest_acc_en** | Configurable (from `TypecastParams`) | Enables 32-bit DEST accumulation; required for UInt32/Int32/Float32 types |
| **preserve_fp32_precision** | Configurable (from `TypecastParams`) | When true, sets `UnpackToDestMode::UnpackToDestFp32` for the input CB, bypassing FP16 truncation during unpack |
| **bfp8_pack_precise** | Configurable (from `TypecastParams`) | Enables precise packing for Bfp8_b output format |
| **unpack_to_dest_mode** | Default or UnpackToDestFp32 | Per-CB setting; FP32 mode preserves full precision through the DEST registers |

#### Hardware Compatibility Notes

- The Wormhole B0 and Blackhole implementations are nearly identical. The primary difference is in address modifier constants used with `TTI_SFPLOAD`/`TTI_SFPSTORE` (e.g., `ADDR_MOD_3` vs `ADDR_MOD_7`, `ADDR_MOD_2` vs `ADDR_MOD_6`). These differences reflect different hardware register mapping conventions but produce the same logical behavior.
- Both architectures support the same set of typecast conversions.
- Some conversions require no SFPU work at all (e.g., Float16_b <-> Bfp8_b, Float16_b <-> Bfp4_b, Float16_b -> Float32). These are handled entirely by the unpacker and/or packer hardware, and the SFPU dispatch body is empty.
- The `_calculate_typecast_*_` functions in the tt-llk submodule contain the actual instruction sequences but are not checked out locally. DeepWiki reports they use `TT_SFPLOADMACRO` with `TTI_SFPNOP` delays, `TTI_SFPSWAP` for clamping, `TTI_SFP_STOCH_RND` for rounding, `TTI_SFPCAST` for integer-float conversion, and `TTI_SFPSHFT2`/`TTI_SFPABS` for sign handling.

## Implementation Notes

1. **Chunk size cap**: The `max_elements_per_chunk` constant of 1024 limits CB memory usage. For narrow rows (<=1024 elements), each row is a single chunk. For wider rows, multiple chunks are needed.

2. **CB page size vs actual transfer size**: The CB page size is always set to `full_chunk_size_bytes`, but partial chunks transfer fewer bytes. This means the last chunk of a row may use less than the full page allocation, wasting some CB space but simplifying the CB configuration (no dynamic page sizes).

3. **No read/write overlap within chunks**: Each chunk read is immediately followed by `noc_async_read_barrier()`, serializing chunk reads. Similarly, writer uses `noc_async_writes_flushed()` per chunk. The double-buffered CBs provide overlap between pipeline stages (reader vs compute vs writer) but not within a single stage.

4. **Cached program reuse**: The `override_runtime_arguments` method updates buffer addresses and row counts without rebuilding the program, enabling efficient reuse across calls with different tensors of the same shape.

5. **Compile-time dtype dispatch**: The `TYPECAST_LLK` and `TYPECAST_LLK_INIT` defines embed the input/output DataFormat values directly into the template parameters, causing the compiler to select exactly one conversion path per kernel compilation. This eliminates all branching at runtime.

6. **Some conversions are SFPU-free**: For format pairs where the unpack/pack pipeline handles the conversion natively (e.g., Float16_b to Bfp8_b), the `typecast_tile` function body is empty -- the data simply passes through DEST unchanged, and the packer reinterprets it in the target format.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the typecast operation work in TTNN? What are the different program factories for typecast?"
   **Reason**: Needed to understand the overall typecast architecture and when the rm_chunked factory is selected.
   **Key Findings**: The rm_chunked factory is selected for ROW_MAJOR layout tensors. Other factories handle interleaved (tiled), sharded, and subgrid cases.

2. **Query**: "How does typecast_tile_init and typecast_tile work in the LLK ckernel layer?"
   **Reason**: Needed to understand the SFPU kernel dispatch mechanism for typecast.
   **Key Findings**: The `_calculate_typecast_*_` functions are in `sfpu/ckernel_sfpu_typecast.h` within the tt-llk submodule. They use SFPU instructions like `TT_SFPLOADMACRO`, `TTI_SFPSWAP`, `TTI_SFP_STOCH_RND`, `TTI_SFPCAST`, `TTI_SFPSHFT2`, `TTI_SFPABS` depending on the conversion direction.

3. **Query**: "How does split_work_to_cores work in tt-metal?"
   **Reason**: Needed to understand core distribution and the two-group pattern.
   **Key Findings**: Returns 6-tuple with num_cores, all_cores, core_group_1, core_group_2, and per-group work counts. Group 1 gets one extra work unit when division is uneven.

4. **Query**: "Show me the full source code of _calculate_typecast_fp32_to_uint16_ and related functions"
   **Reason**: Needed detailed SFPU instruction sequences for the core typecast functions.
   **Key Findings**: FP32->UINT16 uses SFPSWAP (clamp to >=0) + SFP_STOCH_RND. INT32->FP32 uses SFPABS + SFPSHFT2 (sign extract) + SFPCAST. FP32->FP16b uses SFPLOADMACRO + SFPAND.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op_types.hpp`
   **Reason**: Needed to understand TypecastParams fields that configure compute behavior.
   **Key Information**: TypecastParams contains `fp32_dest_acc_en`, `preserve_fp32_precision`, and `bfp8_pack_precise` flags that directly affect SFPU and unpack/pack configuration.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`
   **Reason**: Needed to understand the API-level typecast_tile and typecast_tile_init functions.
   **Key Information**: These are thin wrappers calling `llk_math_eltwise_unary_sfpu_typecast` with template parameters for input/output DataFormat, with `MATH()` macro gating execution to the math RISC only.

3. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h`
   **Reason**: Needed to understand the compile-time dispatch table from DataFormat pairs to specific SFPU kernel functions.
   **Key Information**: Large `if constexpr` chain mapping every supported (in_format, out_format) pair to the corresponding `calculate_typecast_*` function, with several pairs marked as "no SFPU kernel needed, handled by unpacker/packer."
