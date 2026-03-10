# ASIN (Arcsine) SFPU Operation Analysis

## Operation Overview

| Property | Value |
|---|---|
| **Operation Name** | `asin` |
| **Operation Type** | `UnaryOpType::ASIN` |
| **Category** | Elementwise Unary (Trigonometric) |
| **SFPU Function** | `calculate_asin` (Maclaurin series approximation) |
| **Math Approximation Mode** | `false` (always exact, `get_op_approx_mode` returns `false` for all ops by default) |
| **Math Fidelity** | `HiFi4` |
| **Program Factory** | `UnaryProgramFactory` (interleaved) / `UnaryShardedProgramFactory` (sharded) / `UnarySubCoreGridProgramFactory` (sub-core grids) |

The ASIN operation computes the elementwise arcsine (inverse sine) of each element in a tensor. The input domain is [-1, 1]; values outside this range produce NaN. The output range is [-PI/2, PI/2].

## Program Factory Analysis

### Source File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

### Program Factory Selection

The program factory is selected in `UnaryDeviceOperation::select_program_factory`:
- **Sharded input** -> `UnaryShardedProgramFactory`
- **Sub-core grids specified** -> `UnarySubCoreGridProgramFactory`
- **Default (interleaved)** -> `UnaryProgramFactory`

### Factory Structure (UnaryProgramFactory::create)

The `create` method builds a `tt::tt_metal::Program` with three kernels (reader, compute, writer) and the associated circular buffers.

#### Step-by-step flow:

1. **Data format resolution**: Determines input/output data formats and tile sizes from tensor dtypes.
2. **Work splitting**: Calls `split_work_to_cores` to distribute tiles across available compute cores, producing two core groups (group 1 gets `num_pages_per_core_group_1` tiles, group 2 gets `num_pages_per_core_group_2`).
3. **Circular buffer creation**:
   - `c_0` (input): 2 tiles, input data format
   - `c_2` (output): 2 tiles, output data format
   - `c_1` (temporary): only created for HARDSHRINK or LOGIT (not used by ASIN)
4. **Kernel creation**: Reader, writer, and compute kernels are created with appropriate configs.
5. **Runtime args**: Per-core buffer addresses, tile counts, and start IDs are set.

### Circular Buffer Configuration

| CB Index | Purpose | Num Pages | Data Format | Used By |
|---|---|---|---|---|
| `c_0` | Input tiles | 2 | Input tensor dtype | Reader (producer), Compute (consumer) |
| `c_2` | Output tiles | 2 | Output tensor dtype | Compute (producer), Writer (consumer) |

ASIN does not use `c_1` (the temporary buffer). The 2-page double-buffering allows the reader to write the next tile while compute processes the current one.

### Compile-Time Arguments

| Index | Name | Value | Description |
|---|---|---|---|
| 0 | `per_core_block_cnt` | Varies per core | Number of tile blocks to process |
| 1 | `per_core_block_size` | 1 | Tiles per block (always 1 for standard unary) |

### Runtime Arguments

**Compute kernel**: `{packed_scalar1, packed_scalar2}` -- both are 0 for ASIN (no scalar parameters needed).

**Reader kernel**: `{src_buffer_address, num_pages_per_core, start_tile_id}`

**Writer kernel**: `{dst_buffer_address, num_pages_per_core, start_tile_id}`

### Compute Configuration

```cpp
tt::tt_metal::ComputeConfig{
    .math_fidelity = MathFidelity::HiFi4,
    .fp32_dest_acc_en = args.fp32_dest_acc_en,     // Caller-controlled
    .unpack_to_dest_mode = unpack_to_dest_mode,     // Default unless preserve_fp32_precision
    .bfp8_pack_precise = args.bfp8_pack_precise,   // Caller-controlled
    .math_approx_mode = false,                       // All ops default to false
    .compile_args = compute_kernel_args,
    .defines = unary_defines}
```

### Preprocessor Defines

The `unary_defines` map passed to the compute kernel includes:

| Define | Value | Source |
|---|---|---|
| `SFPU_OP_CHAIN_0` | `SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0` | `get_block_defines` |
| `SFPU_OP_CHAIN_0_INIT_0` | `asin_tile_init();` | `get_defines_impl` for `UnaryOpType::ASIN` |
| `SFPU_OP_CHAIN_0_FUNC_0` | `asin_tile(0);` | `get_defines_impl` for `UnaryOpType::ASIN` |
| `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` | `1` | `get_macro_definition` -- ASIN falls to `default` case |
| `INP_FLOAT32` or `INP_FLOAT` | `1` | Based on input dtype |

Note: Unlike operations like SIN, COS, SINH, COSH, ASINH, etc., ASIN does NOT get `SFPU_OP_TRIG_FAMILY_INCLUDE`. It falls through to the `default` case in `get_macro_definition`, which returns `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE`. This is because ASIN's API is defined in `trigonometry.h`, which is always included by the compute kernel directly.

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source buffer address in DRAM
    const uint32_t num_pages = get_arg_val<uint32_t>(1);     // Number of pages (tiles) this core processes
    const uint32_t start_id = get_arg_val<uint32_t>(2);      // Starting tile index for this core

    constexpr auto src_args = TensorAccessorArgs<0>();        // Compile-time tensor accessor config

    constexpr uint32_t cb_id_in0 = 0;                        // CB index c_0 for input

    // Page size is derived from the CB configuration (tile size for TILE layout)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    constexpr uint32_t onepage = 1;                           // Process one page at a time

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);  // Create accessor for source tensor

// Read one tile at a time from DRAM into the input circular buffer
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {           // Forward iteration (default)
#endif
        cb_reserve_back(cb_id_in0, onepage);                  // Wait for space in input CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);   // Get L1 write address
        noc_async_read_page(i, s, l1_write_addr);            // Issue async NoC read from DRAM
        noc_async_read_barrier();                              // Wait for read to complete
        cb_push_back(cb_id_in0, onepage);                     // Signal tile is available to compute
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);      // Destination buffer address in DRAM
    const uint32_t num_pages = get_arg_val<uint32_t>(1);      // Number of pages this core writes
    const uint32_t start_id = get_arg_val<uint32_t>(2);       // Starting tile index

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // Output CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();            // Compile-time tensor accessor config

    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_pages);                      // Sharded: wait for all pages at once
#else

    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {           // Forward iteration (default)
#endif
        cb_wait_front(cb_id_out, onepage);                    // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);     // Get L1 read address
        noc_async_write_page(i, s, l1_read_addr);            // Issue async NoC write to DRAM
        noc_async_writes_flushed();                            // Ensure write is flushed
        cb_pop_front(cb_id_out, onepage);                     // Free the CB slot
    }
    noc_async_write_barrier();                                 // Final barrier for all writes
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

This is the generic SFPU compute kernel used by ASIN and many other unary operations. The specific operation is injected via preprocessor defines (`SFPU_OP_CHAIN_0`).

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"                              // Common compute API (init_sfpu, etc.)
#include "api/compute/tile_move_copy.h"                      // copy_tile API
#include "api/compute/eltwise_unary/eltwise_unary.h"         // Base eltwise unary APIs
#include "api/compute/eltwise_unary/sfpu_split_includes.h"   // Conditional includes based on SFPU_OP_*_INCLUDE defines
#include "api/compute/eltwise_unary/trigonometry.h"          // Trigonometric SFPU ops: asin_tile, asin_tile_init, etc.
#include "api/compute/mul_int_sfpu.h"                        // Integer multiply SFPU
#include "api/compute/eltwise_unary/rpow.h"                  // Reciprocal power
#include "api/compute/eltwise_unary/rdiv.h"                  // Reciprocal division
#include "api/compute/eltwise_unary/fill.h"                  // Fill operation

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of blocks (tiles) to process
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1 for ASIN)

    // Initialize SFPU: configures unpack from c_0, pack to c_2
    // Sets up the LLK math pipeline for SFPU operations
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve space in the output CB for one block of tiles
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to the DST register file
            // This prevents the packer from reading DST while we write to it
            tile_regs_acquire();

            // Wait for the reader to produce one tile in the input CB
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Unpack the tile from input CB into DST register index 0
            // This calls the LLK unpack API to move data from L1 -> source registers -> DST
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // === SFPU_OP_CHAIN_0 expands to: ===
            // asin_tile_init();   -- One-time SFPU initialization (sets up LLK state for asin)
            // asin_tile(0);       -- Compute asin on DST[0], writing result back to DST[0]
#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif

            // Signal that DST is ready for the packer to read
            tile_regs_commit();

            // Wait for the packer to finish with DST (ensures DST is available)
            tile_regs_wait();

            // Pack DST[0] into the output CB
            pack_tile(0, tt::CBIndex::c_2);

            // Free the consumed tile from the input CB
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release DST register file for the next iteration
            tile_regs_release();
        }

        // Push the completed block of output tiles to the writer
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

## SFPU Kernel Implementation

### LLK API Layer

**File**: `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h`

The `asin_tile` and `asin_tile_init` functions are the public API:

```cpp
// asin_tile_init: Initializes the SFPU for arcsine computation.
// Uses SFPU_UNARY_KERNEL_INIT which expands to:
//   llk_math_eltwise_unary_sfpu_init<SfpuType::asin, true>();
// This calls _llk_math_eltwise_unary_sfpu_init_<SfpuType::asin>() which
// configures the math engine's address modifier registers and clears state.
// No custom init callback is needed (unlike atan which needs sfpu_reciprocal_init,
// or sin/cos/tan which need Cody-Waite constant setup).
ALWI void asin_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(asin, true)); }

// asin_tile: Computes arcsine on DST[idst].
// Uses SFPU_UNARY_NO_PARAM_KERNEL_FN which expands to:
//   _llk_math_eltwise_unary_sfpu_params_<true>(
//       ckernel::sfpu::calculate_asin<true>, idst, (int)VectorMode::RC);
// This sets the DST write pointer to idst, calls the SFPU start sequence,
// invokes calculate_asin (the actual math), and calls the done sequence.
ALWI void asin_tile(uint32_t idst) {
    MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_asin, RC, true, idst));
}
```

Key observations:
- ASIN uses `SFPU_UNARY_NO_PARAM_KERNEL_FN` (no extra runtime parameters).
- The `APPROXIMATE` template parameter is hardcoded to `true`, but `calculate_asin` does not use it to select between different approximation levels.
- `VectorMode::RC` means the operation processes all rows and columns of the tile.
- The `MATH(...)` wrapper ensures the code only runs on the TRISC_MATH RISC-V core.

### SFPU Kernel Function

**File** (Blackhole): `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h`
**File** (Wormhole B0): `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h`

Both architectures share identical implementations for ASIN.

#### Annotated SFPU Kernel Source

```cpp
// sfpu_asine_maclaurin_series: Core mathematical kernel for arcsin(x).
// Computes arcsin using a 6-term Maclaurin (Taylor) series expansion around x=0.
// Valid for x in [-1, 1].
//
// Mathematical basis:
// arcsin(x) = x + (1/2)*x^3/3 + (1*3)/(2*4)*x^5/5 + (1*3*5)/(2*4*6)*x^7/7 + ...
// Simplified coefficients:
// arcsin(x) ~ x + (1/6)*x^3 + (3/40)*x^5 + (5/112)*x^7 + (35/1152)*x^9 + (63/2816)*x^11
template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_asine_maclaurin_series(sfpi::vFloat val) {
    sfpi::vFloat tmp = val;                    // Running power of x (x, x^3, x^5, ...)
    sfpi::vFloat val_square = val * val;       // x^2, used to advance powers by 2 each step
    sfpi::vFloat output = tmp;                 // Accumulator, starts with x (first term)

    // Term 2: (1/6) * x^3
    tmp = tmp * val_square;                    // tmp = x^3
    output += 0.166666666 * tmp;               // output += x^3 / 6

    // Term 3: (3/40) * x^5
    tmp = tmp * val_square;                    // tmp = x^5
    output += 0.075 * tmp;                     // output += 3*x^5 / 40

    // Term 4: (5/112) * x^7
    tmp = tmp * val_square;                    // tmp = x^7
    output += 0.044642857 * tmp;               // output += 5*x^7 / 112

    // Term 5: (35/1152) * x^9
    tmp = tmp * val_square;                    // tmp = x^9
    output += 0.03038194 * tmp;                // output += 35*x^9 / 1152

    // Term 6: (63/2816) * x^11
    tmp = tmp * val_square;                    // tmp = x^11
    output += 0.02237216 * tmp;                // output += 63*x^11 / 2816

    return output;
}

// calculate_asin: Top-level SFPU microcode for arcsin.
// Processes ITERATIONS elements (default 8, corresponding to one row of 32 elements
// processed 4 at a time by the SFPU vector lanes = 8 iterations).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_asin() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];     // Read 4 elements from current DST position

        // Domain check: arcsin is only defined for |x| <= 1
        v_if(v < sfpi::vConstNeg1 || v > sfpi::vConst1) {
            // Out-of-domain values produce NaN (IEEE 754 quiet NaN)
            sfpi::dst_reg[0] = std::numeric_limits<float>::quiet_NaN();
        }
        v_else {
            // In-domain values get the Maclaurin series approximation
            sfpi::dst_reg[0] = sfpu_asine_maclaurin_series<APPROXIMATION_MODE>(v);
        }
        v_endif;

        sfpi::dst_reg++;                       // Advance to next group of 4 elements
    }
}
```

### SFPU Instructions Used

The ASIN kernel uses high-level SFPI (SFPU Programming Interface) constructs that compile down to SFPU instructions:

| SFPI Construct | Underlying SFPU Instructions | Description |
|---|---|---|
| `sfpi::dst_reg[0]` (read) | `SFPLOAD` | Load 4 FP32 elements from DST register file |
| `sfpi::dst_reg[0] = ...` (write) | `SFPSTORE` | Store 4 FP32 elements back to DST register file |
| `sfpi::dst_reg++` | Address modifier increment | Advance DST pointer by 4 elements (one SFPU vector width) |
| `val * val` | `SFPMUL` | Vector multiply (4-wide) |
| `output += coeff * tmp` | `SFPMAD` (multiply-add) | Fused multiply-add: output = coeff * tmp + output |
| `v < sfpi::vConstNeg1` | `SFPSETCC` | Set condition codes by comparing v against -1.0 |
| `v > sfpi::vConst1` | `SFPSETCC` | Set condition codes by comparing v against 1.0 |
| `v_if(...) / v_else / v_endif` | `SFPENCC` / `SFPCOMPC` | Enable/disable vector lanes based on condition codes |
| `std::numeric_limits<float>::quiet_NaN()` | `SFPLOADI` + `SFPSTORE` | Load NaN constant and store to masked lanes |

### SFPU Register Usage

- **DST Register File**: The primary data storage. ASIN reads from and writes to `dst_reg[0]` through `dst_reg[7]` (8 iterations, 4 elements each = 32 elements = one row of a 32x32 tile). The `VectorMode::RC` processes all rows and columns.
- **SFPU Local Registers (LREGs)**: Used implicitly for intermediate computations. The polynomial evaluation uses temporary vFloat variables (`tmp`, `val_square`, `output`) which map to SFPU LREGs.
- **Constant Registers**: `sfpi::vConst1` (+1.0) and `sfpi::vConstNeg1` (-1.0) are pre-loaded SFPU constant registers used for the domain check.
- **Condition Code Registers**: Used by `v_if`/`v_else`/`v_endif` for per-lane predication.

### SFPU Execution Flow

1. **Tile Acquisition**: The compute kernel calls `cb_wait_front(c_0, 1)` to wait for the reader to produce one input tile.

2. **Unpack to DST**: `copy_tile(c_0, 0, 0)` invokes the LLK unpack pipeline:
   - The unpacker reads the tile from CB `c_0` in L1 SRAM
   - Data is converted from the source data format (e.g., bfloat16) to FP32 in the source registers
   - Data is copied to DST register index 0

3. **SFPU Init**: `asin_tile_init()` calls `llk_math_eltwise_unary_sfpu_init<SfpuType::asin, true>()`:
   - Calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::asin>()` which configures address modifiers
   - No custom init callback is needed for ASIN (unlike sin/cos which need Cody-Waite constants, or atan which needs reciprocal init)

4. **SFPU Compute**: `asin_tile(0)` triggers the main computation:
   - `_llk_math_eltwise_unary_sfpu_params_<true>` sets the DST write address to tile index 0
   - Calls `_llk_math_eltwise_unary_sfpu_start_` which waits for the math engine
   - Invokes `calculate_asin<true>()` which loops 8 times over the tile:
     - Each iteration processes 4 elements (SFPU vector width)
     - Reads 4 elements from DST
     - Checks if values are in [-1, 1]
     - For in-range values: evaluates 6-term Maclaurin series for arcsin
     - For out-of-range values: writes NaN
     - Advances DST pointer
   - Calls `_llk_math_eltwise_unary_sfpu_done_` which clears the DST address

5. **Pack to Output**: `pack_tile(0, c_2)` invokes the LLK pack pipeline:
   - Reads the result from DST register index 0
   - Converts from FP32 to the output data format
   - Writes to CB `c_2` in L1 SRAM

6. **Buffer Management**: `cb_pop_front(c_0, 1)` frees the input tile; `cb_push_back(c_2, 1)` signals the output tile is ready for the writer.

### SFPU Configuration

| Configuration | Value | Notes |
|---|---|---|
| **Math Fidelity** | `HiFi4` | Highest fidelity mode, no precision shortcuts |
| **Math Approx Mode** | `false` | `get_op_approx_mode` returns false for all ops by default |
| **APPROXIMATION_MODE template** | `true` | Hardcoded in trigonometry.h, but `calculate_asin` does not branch on it |
| **FP32 Dest Acc** | Caller-controlled | When enabled, DST operates in FP32 mode for higher precision |
| **Unpack to Dest Mode** | Default (or FP32 if `preserve_fp32_precision`) | Controls whether unpack converts to FP32 in DST |
| **BFP8 Pack Precise** | Caller-controlled | Enables more precise BFP8 packing |
| **ITERATIONS** | 8 (default) | 8 iterations * 4 SFPU lanes = 32 elements = one tile row |
| **VectorMode** | `RC` (Row+Column) | Processes entire tile |
| **No init callback** | N/A | Unlike sin/cos/tan, ASIN needs no programmable constant setup |

### Hardware Compatibility Notes

**Wormhole B0 vs Blackhole**: The ASIN SFPU kernel implementations are **identical** across both architectures. Both use the same `sfpu_asine_maclaurin_series` function with the same 6-term polynomial and the same `calculate_asin` wrapper.

Key differences in the broader `ckernel_sfpu_trigonometry.h` file (not specific to ASIN but worth noting):
- The `sfpu_tan<true>` (FP32 path) differs: Blackhole uses a quadratic initial reciprocal estimate with `setman`/bitwise manipulation, while Wormhole uses `sfpi::approx_recip` with simpler Newton-Raphson. This difference does not affect ASIN.
- ASIN uses only basic SFPI operations (multiply, multiply-add, load, store, conditional) that are identical across both architectures.

### Mathematical Accuracy Notes

The 6-term Maclaurin series `x + x^3/6 + 3x^5/40 + 5x^7/112 + 35x^9/1152 + 63x^11/2816` converges well for small |x| but has increasing error as |x| approaches 1:

- For |x| < 0.5: relative error is small (< 1e-5)
- For |x| near 1.0: the series converges slowly and the 6-term truncation introduces noticeable error. The true value arcsin(1) = PI/2 ~ 1.5708, but the series gives approximately 1.5708 only with many more terms.

This is a known trade-off: the Maclaurin approach is simple and requires no initialization (no reciprocal, no sqrt), making it cheaper per-element than more sophisticated methods (e.g., Chebyshev or range-reduced polynomial), but less accurate near the domain boundaries.

Note that `acos` (arccos) reuses the same `sfpu_asine_maclaurin_series` with the identity `acos(x) = PI/2 - asin(x)`.

## Python API

The ASIN operation is registered via the `REGISTER_UNARY_OPERATION` macro in `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp`:

```cpp
REGISTER_UNARY_OPERATION(asin, ASIN);
```

This makes `ttnn.asin(input_tensor)` available in Python, which routes through the standard unary operation dispatch path.

## File Index

| File | Role |
|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp` | Program factory (creates Program with reader/compute/writer kernels) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | Generic SFPU compute kernel (SFPU_OP_CHAIN injection point) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | Reader kernel (DRAM -> L1) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | Writer kernel (L1 -> DRAM) |
| `tt_metal/hw/inc/api/compute/eltwise_unary/trigonometry.h` | LLK API: `asin_tile()`, `asin_tile_init()` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` | SFPU kernel (Blackhole): `calculate_asin`, `sfpu_asine_maclaurin_series` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_trigonometry.h` | SFPU kernel (Wormhole B0): identical to Blackhole |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | LLK macro definitions (`SFPU_UNARY_NO_PARAM_KERNEL_FN`, etc.) |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` | LLK init template (`llk_math_eltwise_unary_sfpu_init`) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | Define generation, compute kernel path selection, macro definitions |
| `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | `UnaryOpType::ASIN` enum definition |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | Python API registration (`REGISTER_UNARY_OPERATION(asin, ASIN)`) |

## External Knowledge Sources

### DeepWiki References
- `tenstorrent/tt-metal`: Program factory structure, compute kernel dispatch, circular buffer patterns
- `tenstorrent/tt-llk`: LLK ckernel namespace, SFPU trigonometry implementation details, `_llk_math_eltwise_unary_sfpu_params_` dispatch pattern
- `tenstorrent/tt-metal`: SFPU kernel location (`ckernel_sfpu_trigonometry.h`), `calculate_asin` function, Maclaurin series approach

### Confluence References
Not consulted -- ASIN uses standard SFPI constructs (multiply, multiply-add, conditional execution) that are well-documented in DeepWiki sources.

### Glean References
Not consulted -- the ASIN implementation is straightforward and does not use architecture-specific SFPU instructions that would require confidential hardware specification documents.
