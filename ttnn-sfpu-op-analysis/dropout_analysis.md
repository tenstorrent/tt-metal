# TTNN SFPU Operation Analysis: dropout

## Operation Overview

**Operation Name**: dropout
**Category**: experimental
**Namespace**: `ttnn::experimental::prim`
**SFPU Type**: Eltwise unary (element-wise with PRNG)

**Mathematical Definition**:
For each element x_i in the input tensor, dropout produces:
- `output_i = x_i * scale` with probability `(1 - prob)`
- `output_i = 0` with probability `prob`

The `scale` factor is typically set to `1.0 / (1.0 - prob)` to maintain the expected value of the output during training (inverted dropout).

**Purpose**: Dropout is a regularization technique used during neural network training. It randomly zeroes out a fraction of elements and scales the remaining ones, preventing co-adaptation of neurons and reducing overfitting.

---

## File Inventory

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/experimental/dropout/dropout.hpp` | Public API header declaring `ttnn::experimental::dropout()` |
| `ttnn/cpp/ttnn/operations/experimental/dropout/dropout.cpp` | Public API implementation; delegates to `ttnn::prim::dropout()` |
| `ttnn/cpp/ttnn/operations/experimental/dropout/dropout_nanobind.hpp` | Python binding header |
| `ttnn/cpp/ttnn/operations/experimental/dropout/dropout_nanobind.cpp` | Python binding implementation (nanobind) |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_device_operation_types.hpp` | `DropoutParams` and `DropoutInputs` type definitions |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_device_operation.hpp` | `DropoutDeviceOperation` struct declaration |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_device_operation.cpp` | Device operation implementation (validation, output spec, program hash, factory selection) |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.hpp` | `DropoutProgramFactory` and `DropoutMeshWorkloadFactory` declarations |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/dropout_program_factory.cpp` | Program factory implementation (kernel creation, CB setup, runtime args) |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp` | Compute kernel source (SFPU dispatch) |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp` | Reader dataflow kernel |
| `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp` | Writer dataflow kernel |

### SFPU Kernel Layer Files

| File | Role |
|------|------|
| `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` | Compute API: `dropout_tile()` and `dropout_kernel_init()` |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_dropout.h` | Wormhole B0 LLK wrapper: `calculate_dropout<>()` and `dropout_init<>()` |
| `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_dropout.h` | Blackhole LLK wrapper: `calculate_dropout<>()` and `dropout_init<>()` |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h` | Wormhole B0 SFPU microcode: `_calculate_dropout_()` and `_init_dropout_()` |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h` | Blackhole SFPU microcode: `_calculate_dropout_()` and `_init_dropout_()` |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/ckernel.h` | `init_prng_seed()` definition (Wormhole) |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel.h` | `init_prng_seed()` definition (Blackhole) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` | `_llk_math_eltwise_unary_sfpu_params_()` LLK dispatch |
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` | SFPU macro definitions |

---

## Call Chain

```
ttnn::experimental::dropout()                        [dropout.cpp]
  -> ttnn::prim::dropout()                           [dropout_device_operation.cpp]
    -> ttnn::device_operation::launch<DropoutDeviceOperation>()
      -> DropoutDeviceOperation::select_program_factory()
        -> DropoutProgramFactory (single-device)
        -> DropoutMeshWorkloadFactory (multi-device with per-device seed)
      -> DropoutProgramFactory::create()             [dropout_program_factory.cpp]
        -> CreateCircularBuffer() x2 (c_0 input, c_2 output)
        -> CreateKernel() for reader, writer, compute (group_1 and group_2)
        -> assign_per_core_runtime_args()
```

### Compute Kernel Dispatch Chain
```
dropout_kernel.cpp::kernel_main()
  -> init_sfpu(c_0, c_2)
  -> dropout_kernel_init(seed)                       [dropout.h]
    -> SFPU_ONE_PARAM_KERNEL_INIT(dropout, sfpu::dropout_init, APPROX, seed)
      -> llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, APPROX>(dropout_init<APPROX>, seed)
        -> _init_dropout_(seed)                      [sfpu/ckernel_sfpu_dropout.h]
          -> init_prng_seed(seed)                    [ckernel.h]
  -> dropout_tile(0, int_probability, int_scale_factor)  [dropout.h]
    -> SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(calculate_dropout, RC, APPROX, 0, probability, scale_factor)
      -> _llk_math_eltwise_unary_sfpu_params_<APPROX>(calculate_dropout<APPROX>, 0, RC, probability, scale_factor)
        -> _calculate_dropout_<APPROX, 8>(8, probability, scale)  [sfpu/ckernel_sfpu_dropout.h]
```

---

## Operation Parameters

### DropoutParams (operation_attributes_t)

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_dtype` | `DataType` | Output tensor data type (must match input; hardcoded to `BFLOAT16` in public API) |
| `output_memory_config` | `MemoryConfig` | Output memory configuration |
| `seed` | `uint32_t` | Seed for the hardware PRNG. Passed as a runtime argument so different invocations can use different seeds without cache invalidation. |
| `use_per_device_seed` | `bool` | When `true`, each device in a mesh gets `seed + device_id` as its seed, ensuring different dropout masks per device. Default: `true`. |
| `prob` | `float` | Dropout probability (0.0 to 1.0). Fraction of elements to zero out. |
| `scale` | `float` | Scale factor applied to surviving elements. Typically `1.0 / (1.0 - prob)`. |

### Compile-Time Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `per_core_block_cnt` | `uint32_t` | Number of tile blocks assigned to this core |
| 1 | `per_core_block_dim` | `uint32_t` | Number of tiles per block (always 1 in this implementation) |
| 2 | `int_probability` | `uint32_t` | `(uint32_t)(prob * (double)INT_MAX)` -- integer representation of dropout probability |
| 3 | `int_scale_factor` | `uint32_t` | `std::bit_cast<uint32_t>(scale)` -- bitwise reinterpretation of float scale as uint32 |

### Runtime Arguments (Compute Kernel)

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `seed` | `uint32_t` | PRNG seed; passed at runtime so the program can be cached and reused with different seeds |

### Runtime Arguments (Reader/Writer Kernels)

| Index | Name | Description |
|-------|------|-------------|
| 0 | buffer address | Source/destination buffer address in DRAM |
| 1 | num_tiles | Number of tiles to read/write |
| 2 | start_id | Starting tile index offset |

---

## Circular Buffer Configuration

| CB Index | Name | Data Format | Num Tiles | Role |
|----------|------|-------------|-----------|------|
| `c_0` | Input | Input tensor format | 2 | Double-buffered input: reader writes tiles here, compute reads from here |
| `c_2` | Output | Output tensor format | 2 | Double-buffered output: compute writes results here, writer reads from here |

**Design Notes**:
- Only 2 circular buffers are used because dropout is a simple element-wise unary operation with no auxiliary inputs.
- Double-buffering (2 tiles per CB) allows the reader to write the next tile while the compute kernel processes the current one, enabling pipeline overlap.
- No intermediate CB is needed because the operation is applied in-place on the DST register (copy input -> apply dropout -> pack output).

---

## Work Distribution

The operation uses `split_work_to_cores()` to distribute tiles across the compute grid:
- The compute grid is `compute_with_storage_grid_size` (e.g., 8x8 = 64 cores on Wormhole).
- Tiles are divided evenly, with any remainder assigned to `core_group_1` (which gets one more tile per core than `core_group_2`).
- Two compute kernel instances are created -- one for `core_group_1` and one for `core_group_2` -- because they have different `per_core_block_cnt` compile-time arguments.
- Core iteration uses column-major ordering: `core = {i / num_cores_y, i % num_cores_y}`.

---

## Kernel Implementations

### Reader Kernel

**File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp`

Standard interleaved tile reader using `TensorAccessor`. Reads tiles one at a time from DRAM into CB `c_0` via NoC. Supports both forward (`start_id` to `start_id + num_tiles`) and backward (`start_id` down to `start_id - num_tiles`) iteration via the `BACKWARDS` preprocessor define.

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);   // DRAM buffer address
    uint32_t num_tiles = get_arg_val<uint32_t>(1);   // Total tiles this core processes
    uint32_t start_id = get_arg_val<uint32_t>(2);    // Starting tile index (global offset)

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);  // CB index (c_0)
    constexpr auto src_args = TensorAccessorArgs<1>();            // TensorAccessor compile-time config

    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

// Iterate over tiles; BACKWARDS define reverses direction
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_reserve_back(cb_id_in0, onetile);         // Wait for space in CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s, l1_write_addr);    // DMA tile from DRAM to L1
        noc_async_read_barrier();                     // Wait for DMA completion
        cb_push_back(cb_id_in0, onetile);            // Signal tile is ready for compute
    }
}
```

### Writer Kernel

**File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp`

Standard interleaved tile writer using `TensorAccessor`. Reads result tiles from CB `c_2` and writes them to DRAM via NoC. Also supports `OUT_SHARDED` mode (waits for all tiles in CB without individual write-back) and `BACKWARDS` iteration.

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);    // DRAM buffer address
    uint32_t num_tiles = get_arg_val<uint32_t>(1);    // Total tiles this core writes
    uint32_t start_id = get_arg_val<uint32_t>(2);     // Starting tile index (global offset)

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);  // CB index (c_2)
    constexpr auto dst_args = TensorAccessorArgs<1>();            // TensorAccessor compile-time config

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, num_tiles);  // Sharded: just wait for all tiles, no write-back needed
#else
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_wait_front(cb_id_out, onetile);            // Wait for compute to produce a tile
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(i, s, l1_read_addr);     // DMA tile from L1 to DRAM
        noc_async_write_barrier();                     // Wait for DMA completion
        cb_pop_front(cb_id_out, onetile);             // Free CB slot for compute
    }
#endif
}
```

### Compute Kernel

**File**: `ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp`

This kernel performs the core dropout logic on the SFPU. It initializes the PRNG once, then iterates over all assigned tile blocks. For each tile, it copies input data from CB `c_0` to DST registers, applies the dropout SFPU operation (scale + conditional zero), and packs the result to CB `c_2`.

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/dropout.h"         // Provides dropout_tile() and dropout_kernel_init()
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

void kernel_main() {
    // Compile-time arguments baked into the kernel binary
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // Number of tile blocks to process
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // Tiles per block (always 1)
    uint32_t int_probability = get_compile_time_arg_val(2);     // prob * INT_MAX as uint32
    uint32_t int_scale_factor = get_compile_time_arg_val(3);    // bit_cast<uint32_t>(scale_float)

    // Runtime argument: seed can change per invocation without recompiling
    uint32_t seed = get_arg_val<uint32_t>(0);

    // Initialize the SFPU for unary operation: configure unpack (c_0) and pack (c_2)
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);

    // Initialize the hardware PRNG with the provided seed. This writes to the
    // PRNG_SEED config register and waits ~600 NOPs for the seed to propagate.
    dropout_kernel_init(seed);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // Reserve output space for the entire block dimension before inner loop
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            // Acquire exclusive access to the DST register file (half-sync mode)
            tile_regs_acquire();

            // Wait for one input tile to be available in CB c_0
            cb_wait_front(tt::CBIndex::c_0, 1);

            // Copy tile from CB c_0 (position 0) to DST register (position 0)
            // This invokes the unpacker to move data from L1 to SrcA, then copies to DST
            copy_tile(tt::CBIndex::c_0, 0, 0);

            // Apply dropout on DST[0]: scale surviving elements, zero dropped elements
            // Uses hardware PRNG to generate random values per-element
            dropout_tile(0, int_probability, int_scale_factor);

            // Release DST to the packer (signals that SFPU computation is done)
            tile_regs_commit();

            // Wait for packer to be ready (in half-sync, this waits for the other half)
            tile_regs_wait();

            // Pack tile from DST[0] to CB c_2
            pack_tile(0, tt::CBIndex::c_2);

            // Free the input tile slot in CB c_0
            cb_pop_front(tt::CBIndex::c_0, 1);

            // Release the DST registers for the next iteration
            tile_regs_release();
        }

        // Push the entire block of output tiles to the writer
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
```

---

### SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

#### SFPU Kernel File

- **Wormhole B0**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_dropout.h`
- **Blackhole**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_dropout.h`

Both architectures have identical SFPU microcode for dropout.

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

// probability: integer in range [0, INT_MAX], representing the dropout probability
// scale: uint32_t bitwise representation of a float32 scale factor
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, std::uint32_t probability, std::uint32_t scale)
{
    // --- SFPU microcode begins ---

    // Load the 32-bit scale factor into LREG1 using two 16-bit halves.
    // SFPLOADI with mode 10 loads the low 16 bits; mode 8 loads the high 16 bits.
    // This reconstructs the full float32 scale value in LREG1 across all 32 SIMD lanes.
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);

    // Load the 32-bit probability threshold into LREG2 using the same two-half approach.
    // This is an integer value used for comparison with the PRNG output.
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);

    // Prevent the compiler from unrolling this loop. Each iteration processes one
    // "row" of the current face (4 rows per face x 32 lanes = 128 elements per face).
    // With ITERATIONS=8 and the outer _llk_math_eltwise_unary_sfpu_params_ calling
    // this function 4 times (once per face in RC mode), the full 32x32 tile is covered.
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        ////////////////////////
        // Step 1: Scale samples
        // Load the current destination register value (the input element) into LREG0.
        // SFPLOAD with Mod0=0, AddrMod=3 reads from DST into the LREG.
        ////////////////////////
        TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);

        // Multiply the input value (LREG0) by the scale factor (LREG1).
        // Result goes back to LREG0. This performs: LREG0 = LREG0 * LREG1
        // The third operand LCONST_0 is unused (addend=0 since this is pure multiply).
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        ////////////////////////
        // Step 2: Generate random number
        // SFPMOV with instr_mod1=8 and lreg_c=9 triggers the hardware PRNG.
        // It generates a uint32_t pseudorandom number (32-bit LFSR) per lane
        // and stores it in LREG3.
        ////////////////////////
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);

        // Clear the sign bit of the random number so it is non-negative.
        // This ensures clean unsigned comparison with the probability threshold.
        // SFPSETSGN with instr_mod1=1 sets sign=0 (absolute value).
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        ////////////////////////
        // Step 3: Conditionally drop samples
        // SFPIADD performs integer subtraction: LREG3 = LREG2 - LREG3 (probability - rand).
        // Crucially, instr_mod1=10 means this instruction SETS THE LANE FLAGS
        // based on the result: if LREG2 > LREG3 (i.e., rand < probability), the
        // lane flag is set to true, meaning "this element should be dropped."
        ////////////////////////
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10);

        // SFPMOV with conditional execution: for lanes where the flag is set
        // (rand < probability), move LCONST_0 (zero) into LREG0, zeroing out
        // the scaled value. Lanes where rand >= probability keep their scaled value.
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);

        // Disable conditional execution (clear UseLaneFlagsForLaneEnable).
        // All subsequent instructions will execute on all lanes again.
        TTI_SFPENCC(0, 0, 0, 0);

        // Store LREG0 (either scaled value or zero) back to the DST register.
        // SFPSTORE with Mod0=0, AddrMod=3 writes from LREG to DST.
        TTI_SFPSTORE(0, 0, 3, 0);

        // Advance the DST register pointer to the next row of 32 elements.
        sfpi::dst_reg++;
    }
}

// Initialize the hardware PRNG with a seed value.
inline void _init_dropout_(const std::uint32_t seed)
{
    // Writes the seed to the PRNG_SEED config register and waits for propagation.
    init_prng_seed(seed);
}

} // namespace sfpu
} // namespace ckernel
```

#### SFPU Instructions Used

| Instruction | Operands | Description |
|-------------|----------|-------------|
| `SFPLOADI` | `(LREG_dest, mode, imm16)` | Loads a 16-bit immediate into an LREG. Mode 10 loads low 16 bits; mode 8 loads high 16 bits. Used to reconstruct 32-bit values from two halves. |
| `SFPLOAD` | `(LREG_dest, Mod0, AddrMod, imm10)` | Loads data from the DST register file into an LREG. Bridges the DST data format to LREG's 32-bit float format. |
| `SFPMUL` | `(VA, VB, VC_addend, VD_dest, mod)` | Floating-point multiply: `VD = VA * VB`. The VC addend is LCONST_0 (zero), so this is a pure multiply. |
| `SFPMOV` (PRNG) | `(imm12, lreg_c=9, LREG_dest, instr_mod1=8)` | When `lreg_c=9` and `instr_mod1=8`, generates a pseudorandom uint32 per lane using the hardware 32-bit LFSR PRNG and stores it in LREG_dest. |
| `SFPMOV` (cond) | `(imm12, LCONST_0, LREG_dest, 0)` | Conditional move: only executes on lanes where the lane flag is set. Moves zero (LCONST_0) into LREG_dest to zero out dropped elements. |
| `SFPSETSGN` | `(imm, LREG_src, LREG_dest, mod=1)` | Sets the sign bit. With `mod=1`, clears the sign bit (makes value non-negative / absolute value). Used to make the random number unsigned for comparison. |
| `SFPIADD` | `(imm, LREG_A, LREG_B, instr_mod1=10)` | Integer addition/subtraction with flag setting. Computes `LREG_A - LREG_B` and sets per-lane flags based on the result. `instr_mod1=10` enables flag setting. |
| `SFPENCC` | `(imm2, VD, mod1, ...)` | Enables/disables conditional execution. `SFPENCC(0,0,0,0)` clears `UseLaneFlagsForLaneEnable`, returning all lanes to unconditional execution. |
| `SFPSTORE` | `(LREG_src, Mod0, AddrMod, imm10)` | Stores data from an LREG back to the DST register file, converting from LREG format to DST format. |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| `LREG0` | Working register: holds the current input element, then the scaled value, then the final result (scaled or zero) |
| `LREG1` | Holds the `scale` factor (float32 reconstructed from two 16-bit immediates). Loaded once before the loop. |
| `LREG2` | Holds the `probability` threshold (uint32 reconstructed from two 16-bit immediates). Loaded once before the loop. |
| `LREG3` | Holds the hardware-generated pseudorandom number (uint32). Regenerated each iteration. |
| `LCONST_0` | Pre-defined constant register holding 0.0. Used as the "drop" value and as the zero addend in SFPMUL. |
| `dst_reg` | DST register file pointer. Incremented each iteration to process the next row of 32 elements. |
| `PRNG_SEED` | Config register written by `init_prng_seed()`. Seeds the hardware LFSR for random number generation. |

#### SFPU Execution Flow

1. **Initialization** (`dropout_kernel_init`):
   - Calls `_init_dropout_(seed)`, which calls `init_prng_seed(seed)`.
   - `init_prng_seed` writes the seed to the `PRNG_SEED` configuration register and executes 600 `SFPNOP` instructions to allow the seed to propagate through the LFSR hardware.

2. **Per-tile processing** (outer loop in compute kernel):
   - `cb_wait_front(c_0, 1)`: Wait for reader to deliver one input tile.
   - `copy_tile(c_0, 0, 0)`: Unpack tile from CB `c_0` into DST register slot 0.
   - `dropout_tile(0, probability, scale_factor)`: Invoke the SFPU dropout operation.

3. **SFPU dispatch** (`_llk_math_eltwise_unary_sfpu_params_`):
   - Sets the DST write address to tile index 0.
   - Stalls until the math pipeline is idle (`STALL_SFPU, MATH`).
   - In `VectorMode::RC` mode, calls `_calculate_dropout_` 4 times (once per face of the 32x32 tile).
   - Between faces, advances the DST row counter via `SETRWC` instructions (2 advances of 8 rows each per face).

4. **Per-face SFPU execution** (`_calculate_dropout_` with `ITERATIONS=8`):
   - Loads `scale` into LREG1 and `probability` into LREG2 (4 SFPLOADI instructions, done once before the loop).
   - For each of the 8 rows in the face:
     a. `SFPLOAD`: Load current DST value into LREG0.
     b. `SFPMUL`: Multiply LREG0 by LREG1 (scale the element).
     c. `SFPMOV(PRNG)`: Generate random uint32 per lane into LREG3.
     d. `SFPSETSGN`: Clear sign bit of LREG3 (make non-negative for comparison).
     e. `SFPIADD`: Compute `probability - rand`, set lane flags (flag=true if rand < probability).
     f. `SFPMOV(cond)`: For flagged lanes, overwrite LREG0 with zero.
     g. `SFPENCC`: Disable conditional execution.
     h. `SFPSTORE`: Write LREG0 back to DST.
     i. `dst_reg++`: Advance to next row.

5. **Post-SFPU** (back in compute kernel):
   - `tile_regs_commit()`: Signal DST is ready for packing.
   - `tile_regs_wait()`: Wait for packer readiness.
   - `pack_tile(0, c_2)`: Pack DST[0] to CB `c_2`.
   - `cb_pop_front(c_0, 1)`: Free the input CB slot.
   - `tile_regs_release()`: Release DST for next iteration.

6. **Block completion**:
   - `cb_push_back(c_2, per_core_block_dim)`: Signal writer that output tiles are ready.

#### SFPU Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| `MathFidelity` | `HiFi4` | Highest fidelity. Dropout does not benefit from reduced fidelity since the math is just multiply + conditional zero. |
| `fp32_dest_acc_en` | `false` | 16-bit DST accumulation is sufficient; dropout operates on bfloat16 data. |
| `math_approx_mode` | `false` | No approximation needed. The SFPU instructions used (multiply, compare, conditional move) are exact operations. The `APPROXIMATION_MODE` template parameter is passed through but not actually used in `_calculate_dropout_`. |
| `VectorMode` | `RC` | Process all 4 faces of the 32x32 tile (full rows and columns). |
| `ITERATIONS` | `8` | Default value. Each call to `_calculate_dropout_` processes 8 rows of 32 elements = 256 elements per face. With 4 faces, that covers the full 1024 elements (32x32 tile). |
| `per_core_block_dim` | `1` | One tile per block. The outer loop in the compute kernel iterates `per_core_block_cnt` times, processing one tile per iteration. |

#### Hardware Compatibility Notes

- **Wormhole B0 vs Blackhole**: The SFPU microcode for dropout is **identical** between both architectures. The `_calculate_dropout_` and `_init_dropout_` functions in `tt_llk_wormhole_b0` and `tt_llk_blackhole` are byte-for-byte the same.
- **Blackhole includes** `sfpi_fp16.h` in addition to the common includes, but this header is not used by the dropout kernel.
- The Wormhole B0 LLK wrapper (`ckernel_sfpu_dropout.h`) additionally includes `ckernel.h` and `ckernel_defs.h`, while the Blackhole version relies on indirect inclusion through `sfpu/ckernel_sfpu_dropout.h`.
- The hardware PRNG (`SFPMOV` with `lreg_c=9, instr_mod1=8`) uses a 32-bit LFSR. DeepWiki notes this has "poor statistical properties" -- the randomness quality is sufficient for dropout regularization but not for cryptographic or high-quality statistical purposes.
- The `init_prng_seed` function is identical on both architectures: it writes the seed to `PRNG_SEED_Seed_Val_ADDR32` and waits 600 NOP cycles for propagation.

---

## Program Factory Design

### Single-Device vs Multi-Device

The operation supports two program factory variants:

1. **`DropoutProgramFactory`**: Used when `use_per_device_seed = false`. Creates a single program with the same seed on all devices.
2. **`DropoutMeshWorkloadFactory`**: Used when `use_per_device_seed = true` (default). Creates a separate program per device in the mesh, with each device's seed offset by its device ID (`seed + device_id`). This ensures each device generates a different dropout mask, which is critical for correct distributed training.

The factory selection happens in `DropoutDeviceOperation::select_program_factory()`.

### Program Cache Design

The `compute_program_hash` function excludes the `seed` from the hash because the seed is a runtime argument that changes every forward pass. This allows the compiled program to be cached and reused across invocations with different seeds, avoiding recompilation overhead. The hash includes: `prob`, `scale`, `output_dtype`, `output_memory_config`, the program factory variant index, input dtype, memory config, and padded volume.

### Shared Variables

The `DropoutSharedVariables` struct caches kernel handles and core group information for use by `override_runtime_arguments`. On cache hits, only the seed and buffer addresses need updating -- the tile counts and core assignments remain stable.

---

## Validation Rules

From `validate_on_program_cache_miss`:

1. Input and output data types must match.
2. Input must be on device (`StorageType::DEVICE`) with a non-null buffer.
3. Input and output memory layouts must match.
4. For non-sharded inputs: tensor must be in `TILE` layout with `INTERLEAVED` memory.
5. For preallocated output: shape must match computed output shape; if non-sharded, must be in `TILE` layout.

---

## Python API

```python
ttnn.experimental.dropout(
    input_tensor,       # ttnn.Tensor (BFLOAT16, TILE layout)
    probability=0.2,    # float: dropout probability
    scale=1.25,         # float: typically 1/(1-prob)
    seed=42,            # uint32_t: PRNG seed
    use_per_device_seed=True,   # bool: per-device seed offset
    memory_config=None,         # optional ttnn.MemoryConfig
    output_tensor=None          # optional preallocated output
)
```

**Supported**: BFLOAT16 dtype, TILE layout, ranks 2/3/4.

---

## External Knowledge Sources

### DeepWiki References

- **tenstorrent/tt-metal**: Dropout operation architecture, program factory pattern, `DropoutDeviceOperation` structure, compute kernel setup.
- **tenstorrent/tt-llk**: `ckernel::sfpu::_calculate_dropout_` implementation details, PRNG via `SFPMOV`, LFSR characteristics, `init_prng_seed` function, LLK API dispatch pattern (`_llk_math_eltwise_unary_sfpu_params_`).
- **tenstorrent/tt-isa-documentation**: SFPU instruction semantics for `SFPLOADI`, `SFPLOAD`, `SFPMUL`, `SFPMOV`, `SFPSETSGN`, `SFPIADD`, `SFPENCC`, `SFPSTORE`; lane flag mechanism; conditional execution model; PRNG LFSR details.
- **tenstorrent/sfpi**: `dst_reg` abstraction, LREG registers, `LCONST_0` constant, `init_prng_seed` seed propagation.

### Confluence References

Not consulted. DeepWiki provided sufficient detail on all SFPU instructions used in the dropout kernel.

### Glean References

Not consulted. The dropout SFPU implementation is fully documented in open-source sources.
