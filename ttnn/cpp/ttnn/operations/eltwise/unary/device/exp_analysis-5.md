# EXP Operation Implementation Analysis

## Overview

The EXP operation computes the element-wise natural exponential function `e^x` for every element of an input tensor. It is implemented as a unary SFPU operation using the generic `UnaryProgramFactory` (and its `UnarySubCoreGridProgramFactory` variant), which also serves as the backbone for many other unary element-wise operations (relu, log, sqrt, etc.).

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

The EXP operation is registered via `UnaryOpType::EXP`. The `fast_and_approximate_mode` boolean parameter (stored as param0) controls which SFPU algorithm is used.

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `input.buffer()->num_pages()` (total tiles in the tensor) |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for standard factory) |

In the standard `UnaryProgramFactory`, each work unit is exactly 1 tile. The compute kernel processes one tile at a time: wait for reader to produce it, copy to DEST, apply SFPU exp, pack result, push to writer.

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | Flattened to pages |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, or INT32/UINT32 (type-dependent code paths) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (may differ for BITCAST, not applicable to EXP) |

### Layout Transformations
None. The EXP operation preserves layout and data format. The operation is shape-agnostic; it treats the tensor as a flat sequence of pages (tiles).

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)`, SFPU exp, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

The reader reads one tile at a time into CB c_0. The compute kernel processes tiles one-by-one within each block, but batches the `cb_push_back` for the entire block dimension (which is 1 for the standard factory). The writer drains one tile at a time from CB c_2.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Notes:
- CB c_1 (tmp0) is only created for HARDSHRINK, CBRT, or LOGIT operations, NOT for EXP.
- Both input and output CBs are double-buffered (capacity = 2 * page_size), enabling the reader to fill one slot while compute processes the other.
- For BITCAST operations, the input CB uses the output data format; this is not applicable to EXP.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity of 2 tiles and block size of 1 tile, enabling double-buffering. This allows:
- Reader and compute to overlap: reader fills the next tile while compute processes the current one.
- Compute and writer to overlap: compute produces the next tile while writer drains the current one.

## Index Calculations

The reader and writer use `TensorAccessor` for index-to-address mapping. Each core is assigned a contiguous range of page indices:
- `start_id = num_pages_written` (cumulative sum of pages assigned to prior cores)
- Pages processed: `[start_id, start_id + num_pages_per_core)`

`noc_async_read_page(i, s, l1_write_addr)` translates page index `i` to a physical DRAM address via the TensorAccessor, which handles the interleaved bank mapping internally.

## Memory Access Patterns

### Read Pattern
**Sequential**: Pages are read in order from `start_id` to `start_id + num_pages - 1`. Each page is read via a single NoC read (`noc_async_read_page`), followed by a barrier. The access is page-granular with interleaved banking handled by the TensorAccessor.

### Write Pattern
**Sequential**: Pages are written in the same order as they were read. Each page is written via `noc_async_write_page`, with a final `noc_async_write_barrier()` after all pages.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (flattened from 2D grid) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8 on Wormhole) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two core groups: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

The `split_work_to_cores` utility divides total pages evenly across the available compute grid. Cores are enumerated in column-major order (`core = {i / num_cores_y, i % num_cores_y}`). The two-group approach handles remainder tiles: if `num_pages` is not evenly divisible, group 1 cores each get one extra tile.

Two separate compute kernels are created for the two core groups, differing only in their `per_core_block_cnt` compile-time argument.

## Arguments

### Compile-Time Arguments

**Reader Kernel** (`reader_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for src_buffer (buffer type, address mapping) |

**Writer Kernel** (`writer_unary_interleaved_start_id.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output CB index (c_2 = 2) |
| 1+ | TensorAccessorArgs | uint32_t[] | Packed tensor accessor parameters for dst_buffer |

**Compute Kernel** (`eltwise_sfpu.cpp`):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for standard factory) |

### Runtime Arguments

**Reader Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core reads |
| 2 | start_id | uint32_t | Starting page index for this core |

**Writer Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM/L1 address |
| 1 | num_pages | uint32_t | Number of pages this core writes |
| 2 | start_id | uint32_t | Starting page index for this core |

**Compute Kernel**:

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for EXP (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for EXP (always 0) |

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader | RISCV_0 | NOC0 | DRAM/L1 | CB c_0 | Read tiles via TensorAccessor |
| compute | RISCV_2 (math) | N/A | CB c_0 | CB c_2 | init_sfpu, copy_tile, exp_tile, pack_tile |
| writer | RISCV_1 | NOC1 | CB c_2 | DRAM/L1 | Write tiles via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential page reader. Creates a TensorAccessor from compile-time args, then loops from `start_id` to `end_id`, reading one page at a time into CB c_0. Supports `BACKWARDS` define for reverse iteration (not used by EXP).

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Sequential page writer. Waits for compute to produce one tile in CB c_2, reads its L1 address, writes to DRAM via NoC, pops the tile. Also supports `OUT_SHARDED` define (not used by default EXP). Final `noc_async_write_barrier()` ensures all writes complete.

### Compute Kernel

#### Compute Kernel File
`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`

#### Annotated Compute Kernel Source

```cpp
// SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"  // conditionally includes exp.h when SFPU_OP_EXP_INCLUDE is defined
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);  // number of tiles this core processes
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);  // tiles per block (1 for standard factory)

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);  // configures unpacker (c_0 format) and packer (c_2 format) for SFPU path
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);  // reserve output space for the entire block dimension
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();  // acquire exclusive access to DEST registers for math RISC-V

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);  // blocks until reader has produced at least 1 tile in c_0

            copy_tile(tt::CBIndex::c_0, 0, 0);  // unpacks tile 0 from CB c_0 into DEST register 0

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0  // expands to: SFPU_OP_CHAIN_0_INIT_0 SFPU_OP_CHAIN_0_FUNC_0
            // For EXP with param (fast_and_approx):
            //   SFPU_OP_CHAIN_0_INIT_0 -> exp_tile_init<param0>();
            //   SFPU_OP_CHAIN_0_FUNC_0 -> exp_tile<param0>(0);
            // For EXP without param:
            //   SFPU_OP_CHAIN_0_INIT_0 -> exp_tile_init();
            //   SFPU_OP_CHAIN_0_FUNC_0 -> exp_tile(0);
#endif

            tile_regs_commit();  // transfers DEST register ownership from math to pack RISC-V

            tile_regs_wait();  // pack RISC-V waits for math to commit

            pack_tile(0, tt::CBIndex::c_2);  // packs DEST[0] result into CB c_2 output buffer

            cb_pop_front(tt::CBIndex::c_0, 1);  // frees the consumed tile in c_0 so reader can reuse the slot

            tile_regs_release();  // releases DEST registers back to math RISC-V for the next iteration
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);  // publishes the block to the writer
    }
}
```

### SFPU Kernel Implementation

The EXP operation has multiple SFPU implementation paths, selected by template parameters at compile time. The dispatch chain is:

1. `exp_tile<approx, fast_and_approx, ...>(idst)` in `api/compute/eltwise_unary/exp.h`
2. Calls `calculate_exponential<APPROXIMATION_MODE, FAST_APPROX, is_fp32_dest_acc_en, ...>()` in `hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
3. Dispatches to one of:
   - **FAST_APPROX && APPROX && CLAMP_NEGATIVE**: SFPLOADMACRO-based Schraudolph algorithm with input clamping (default for `fast_and_approximate_mode=true`)
   - **FAST_APPROX && APPROX && !CLAMP_NEGATIVE**: Replay-buffer-based Schraudolph algorithm with SFPSHFT2 + SETSGN
   - **APPROX && !FAST_APPROX**: Piecewise approximation via `_calculate_exponential_piecewise_`
   - **!APPROX && !fp32_dest_acc**: `_sfpu_exp_21f_` (Moroz 2022 exp_21f polynomial)
   - **!APPROX && fp32_dest_acc**: `_sfpu_exp_f32_accurate_` (Cody-Waite range reduction + 7th order Taylor)

#### SFPU Kernel File (arch-specific)
- **Blackhole**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Wormhole**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Shared (LLK)**: `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_exp.h`

The arch-specific files (`hw/ckernels/`) contain `calculate_exponential` (the top-level dispatch), `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, and `_sfpu_exp_f32_accurate_`. The shared LLK files (`third_party/tt_llk/`) contain `_sfpu_exp_`, `_calculate_exponential_body_`, `_calculate_exponential_approx_`, `_calculate_exponential_piecewise_`, `_calculate_exponential_`, and `_init_exponential_`. The Wormhole and Blackhole implementations of the shared LLK file are **identical** in structure.

#### Annotated SFPU Kernel Source (Shared LLK — `_calculate_exponential_` and `_init_exponential_`)

```cpp
// This is the main dispatch function called by calculate_exponential when APPROXIMATION_MODE is true
template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS, bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE = true>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor /* 1.0f in BF16 */)
{
    if constexpr (FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE)
    {
        // === Path 1: SFPLOADMACRO-based Schraudolph with input clamping ===
        // This is the default path when fast_and_approximate_mode=true and InputClamping::ClampToNegative

        // Phase 1: Sanitize inputs — clamp values below -88.5 to -88.5 using SFPSWAP macro
        // Uses Macro Sequence Register 1 (sanitize): LD → SWAP → STORE
        // SFPSWAP compares loaded value against LREG[14] (-88.5) and keeps the larger
        // 8 LOADMACRO calls cover all 16 dest offsets (0,2,4,6,8,10,12,14) = 8 faces of the tile
        // Each LOADMACRO uses one of LREG[0-3] as the working register, cycling through them
        TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 0);   // Seq 1: sanitize face at dest offset 0 via LREG[0]
        TTI_SFPNOP;                                // NOP: SWAP takes 2 cycles, not pipelined
        TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 2);   // Seq 1: sanitize face at dest offset 2 via LREG[1]
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 4);   // ... dest offset 4 via LREG[2]
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 6);   // ... dest offset 6 via LREG[3]
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 8);   // ... dest offset 8 via LREG[0]
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 10);  // ... dest offset 10 via LREG[1]
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(6, 0, ADDR_MOD_7, 12);  // ... dest offset 12 via LREG[2]
        TTI_SFPNOP;
        TTI_SFPLOADMACRO(7, 0, ADDR_MOD_7, 14);  // ... dest offset 14 via LREG[3]
        // No NOP needed here because the next macro is computational and doesn't immediately use SIMPLE unit

        // Phase 2: Compute exp via Macro Sequence Register 0
        // Sequence: LD → MAD → ROUND → SHIFT → STORE
        //   MAD:   i = A * x + (B-C)    where A = 256/ln2, B-C = 32500.818
        //   ROUND: convert fp32 to uint16 via stochastic rounding
        //   SHIFT: left-shift by 15 to place integer bits in IEEE754 exponent field
        // 8 LOADMACRO calls process all 16 faces (even/odd columns, rows 0-15)
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);   // Seq 0: compute exp for face at dest offset 0
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);   // ... dest offset 2
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 4);   // ... dest offset 4
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 6);   // ... dest offset 6
        TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 8);   // ... dest offset 8
        TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 10);  // ... dest offset 10
        TTI_SFPLOADMACRO(2, 0, ADDR_MOD_7, 12);  // ... dest offset 12
        TTI_SFPLOADMACRO(3, 0, ADDR_MOD_7, 14);  // ... dest offset 14
        TTI_SFPNOP;  // drain: wait for final macro to complete
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 8)
    {
        // === Path 2: Replay-buffer Schraudolph with SETSGN (no input clamping) ===
        // Uses replay buffer for efficiency: ~2.5 cycles/element
        // Pattern: 16 instructions (8 LM + 8 SHFT2), replayed once
        // SFPSHFT2 left-shifts by 15 (from LREG[14]) and writes to LREG[4]
        // SETSGN restores the sign from the STOCHRND result

        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},  // auto-increment dest by 2 per LOADMACRO
        }.set(ADDR_MOD_7);

        lltt::replay(0, 16);  // replay the 32-instruction pattern recorded during init

        // Drain phase: final 2 SHFT2 operations
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[6]
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);  // SHFT2[7]
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else if constexpr (FAST_APPROX && APPROXIMATION_MODE && ITERATIONS == 32)
    {
        // === Path 3: 32-element replay-buffer version ===
        // Same as Path 2 but for 32 elements: ~2.125 cycles/element
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }.set(ADDR_MOD_7);

        lltt::replay(0, 32);
        lltt::replay(0, 32);

        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    else
    {
        // === Path 4: SFPI-based (non-fast-approx or non-approx) ===
        // Processes one face at a time using SFPI vector registers
        for (int d = 0; d < ITERATIONS; d++)
        {
            sfpi::vFloat val    = sfpi::dst_reg[0];  // load face element from DEST register
            sfpi::vFloat result = _calculate_exponential_piecewise_<APPROXIMATION_MODE, SCALE_EN, SKIP_POSITIVE_CHECK>(val, exp_base_scale_factor);
            sfpi::dst_reg[0]    = result;  // store result back to DEST
            sfpi::dst_reg++;  // advance to next face in the tile
        }
    }
}
```

#### Annotated SFPU Kernel Source (Arch-specific — `calculate_exponential` and improved implementations)

```cpp
// Top-level dispatch in arch-specific ckernel_sfpu_exp.h
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool is_fp32_dest_acc_en,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false,
    bool CLAMP_NEGATIVE = true>
void calculate_exponential(const uint exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    if constexpr (APPROXIMATION_MODE) {
        // Delegate to shared LLK implementation (LOADMACRO or piecewise path)
        _calculate_exponential_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK,
            CLAMP_NEGATIVE>(exp_base_scale_factor);
    } else {
        // Non-approximate mode: use improved polynomial algorithms
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];  // load from DEST
            if constexpr (SCALE_EN) {
                val = val * sfpi::s2vFloat16b(exp_base_scale_factor);  // optional input scaling
            }
            sfpi::vFloat result = _sfpu_exp_improved_<is_fp32_dest_acc_en>(val);
            // Dispatches to:
            //   _sfpu_exp_21f_<false>(val) when fp32_dest_acc is off (bfloat16 path)
            //   _sfpu_exp_f32_accurate_(val) when fp32_dest_acc is on (float32 path)
            sfpi::dst_reg[0] = result;  // store back
            sfpi::dst_reg++;  // advance to next face
        }
    }
}

// === _sfpu_exp_21f_: Moroz 2022 "exp_21f" algorithm ===
// Computes exp(x) = 2^(x/ln2) by decomposing x/ln2 into integer and fractional parts
// Uses a 2nd degree polynomial for the fractional part approximation
// Accuracy: ~21 significant bits of float (adequate for bfloat16)
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    sfpi::vFloat xlog2 = (val * ONE_LN2 + 127.f);  // scale to IEEE754 exponent+bias range

    // Clamp to [0, 255] to prevent overflow/underflow
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);  // xlog2 = max(xlog2, 0)
    sfpi::vec_min_max(xlog2, threshold_high);  // xlog2 = min(xlog2, 255)

    sfpi::vInt z = _float_to_int32_for_exp21f_(xlog2);  // convert to fixed-point (scaled by 2^23)

    sfpi::vInt exponential_part = exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(z));  // extract exponent bits = 2^(integer part)
    sfpi::vInt fractional_part = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));     // extract 9-bit mantissa = fractional part

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, 0);  // convert mantissa to float

    // 2nd degree polynomial: 2^(x_f) ~ c0 + c1*x_f + c2*x_f^2 (on range [0, 2^23])
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: result = 2^(x_i) * 2^(x_f) by setting exponent on the polynomial result
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    if constexpr (!is_fp32_dest_acc_en) {
        // Explicit round-to-nearest-even conversion to bfloat16 to avoid truncation artifacts
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }

    return y;
}

// === _sfpu_exp_f32_accurate_: Cody-Waite range reduction + Taylor series ===
// Target accuracy: < 1 ULP for float32
sfpi_inline sfpi::vFloat _sfpu_exp_f32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;
    constexpr float INV_LN2 = 1.4426950408889634f;

    sfpi::vFloat z = val * INV_LN2;  // z = x / ln(2)

    sfpi::vInt exp_bits = sfpi::exexp(z);  // extract exponent for NaN detection

    v_if(z >= OVERFLOW_THRESHOLD) {
        result = std::numeric_limits<float>::infinity();  // overflow -> +inf
    }
    v_elseif(z <= UNDERFLOW_THRESHOLD) {
        result = sfpi::vConst0;  // underflow -> 0
    }
    v_elseif(exp_bits == 255) {
        result = std::numeric_limits<float>::quiet_NaN();  // NaN passthrough
    }
    v_else {
        sfpi::vInt k_int;
        sfpi::vFloat k = _sfpu_round_nearest_int32_(z, k_int);  // k = round(x/ln2)

        // Cody-Waite range reduction: r = x - k*ln(2) in extended precision
        constexpr float LN2_HI = -0.6931152343750000f;   // negated for SFPMAD optimization
        constexpr float LN2_LO = -3.19461832987e-05f;

        sfpi::vFloat r_hi = k * LN2_HI + val;  // compiles to single SFPMAD
        sfpi::vFloat r = k * LN2_LO + r_hi;    // compiles to single SFPMAD

        // 7th order Taylor polynomial: exp(r) ~ 1 + r + r^2/2! + ... + r^7/7!
        sfpi::vFloat p = PolynomialEvaluator::eval(
            r, sfpi::vConst1, sfpi::vConst1, 0.5f,
            1.0f/6.0f, 1.0f/24.0f, 1.0f/120.0f, 1.0f/720.0f, 1.0f/5040.0f);

        // Scale by 2^k: add k to the exponent of p
        sfpi::vInt p_exp = sfpi::exexp_nodebias(p);
        sfpi::vInt new_exp = p_exp + k_int;
        result = sfpi::setexp(p, new_exp);
    }
    v_endif;

    return result;
}
```

#### SFPU Instructions Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::dst_reg[n]` | Read/write lane elements from/to DEST register at offset n |
| `sfpi::dst_reg++` | Advance DEST register pointer to next face (row pair) |
| `sfpi::exexp(v)` | Extract biased exponent from IEEE754 float |
| `sfpi::exexp_nodebias(v)` | Extract raw (unbiased) exponent bits |
| `sfpi::exman8(v)` | Extract 8-bit mantissa with implicit leading 1 |
| `sfpi::exman9(v)` | Extract 9-bit mantissa |
| `sfpi::setexp(v, e)` | Set the exponent of float v to e (used for 2^k scaling) |
| `sfpi::setsgn(v, s)` | Set sign bit of float v (used to force positive for exp) |
| `sfpi::shft(v, n)` | Barrel shift by n bits |
| `sfpi::addexp(v, n)` | Add n to the exponent of v (multiply by 2^n) |
| `sfpi::vec_min_max(a, b)` | Simultaneously compute min and max of a and b |
| `sfpi::int32_to_float(v, mode)` | Convert integer to float |
| `sfpi::float_to_fp16b(v, mode)` | Convert float32 to bfloat16 with rounding |
| `sfpi::s2vFloat16b(v)` | Scalar to vector bfloat16 broadcast |
| `sfpi::reinterpret<T>(v)` | Bitwise reinterpret between vFloat/vInt/vUInt |
| `PolynomialEvaluator::eval(x, c0, c1, ...)` | Evaluate polynomial c0 + c1*x + c2*x^2 + ... in Horner form |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution (condition codes) |
| `v_and(cond)` | Narrow predication within a `v_if` block |
| `_sfpu_reciprocal_<2>(v)` | 2-iteration Newton-Raphson reciprocal (used in non-approx path for negative inputs) |
| `TTI_SFPLOADMACRO(lreg, seq, addr_mod, dest)` | Execute macro sequence: LD + configured pipeline stages |
| `TTI_SFPMAD(a, b, c, d, mod)` | Multiply-accumulate: d = a * b + c (also used for backdoor macro loading) |
| `TTI_SFP_STOCH_RND(...)` | Stochastic rounding: fp32 to int16 conversion |
| `TTI_SFPSHFT(imm, c, dest, mod)` | Barrel shift by immediate |
| `TTI_SFPSHFT2(a, b, c, mod)` | Barrel shift from register (mode 5: shift amount from VC) |
| `TTI_SFPSETSGN(imm, c, dest, mod)` | Set sign bit from source register |
| `TTI_SFPLOADI(lreg, mode, imm)` | Load immediate into LREG (lo16/hi16 halves) |
| `TTI_SFPCONFIG(val, dest, mode)` | Configure SFPU: load LREG constants, macro instruction/sequence registers |
| `TTI_SFPNOP` | SFPU no-operation (pipeline drain/timing) |
| `lltt::replay(start, count)` | Replay recorded instruction buffer |
| `lltt::record(start, count)` | Record instructions into replay buffer |

#### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[0..15]** | Tile face data: 16 dest offsets, each covering 2 columns x 4 rows of a 32x32 tile. Read by SFPLOAD, written by SFPSTORE |
| **LREG[0-3]** | Working registers used by SFPLOADMACRO for per-element computation |
| **LREG[4]** | Staging register for SFPSHFT2 output (non-clamping path) |
| **LREG[12]** | Constant A = 256/ln(2) = 369.33 (Schraudolph coefficient) |
| **LREG[13]** | Constant B-C = 32500.818 (Schraudolph bias minus correction) |
| **LREG[14]** | Threshold -88.5 (clamping path) or shift amount 15 (non-clamping path) |
| **LREG[16]** | Staging register for SETSGN output (non-clamping path) |
| **vConstFloatPrgm0** | 1/ln(2) = 1.442695 (slow approx path) |
| **vConstFloatPrgm1** | C23_73 constant (slow approx path) |
| **vConstIntPrgm2** | ADJ_EXP adjustment (slow approx path) |

#### SFPU Execution Flow

1. **Initialization** (`exp_init` / `_init_exponential_`):
   - For FAST_APPROX + CLAMP: Loads constants (-88.5, A, B-C) into LREGs, programs 4 macro instruction registers (SWAP, MAD, STOCH_RND, SHFT), configures 2 macro sequence registers (sanitize + compute)
   - For FAST_APPROX + !CLAMP: Loads constants (A, B-C, shift=15), programs 3 macro instructions (MAD, STOCH_RND, SETSGN), configures 1 macro sequence, records 32-instruction replay buffer pattern
   - For APPROX + !FAST: Sets programmable constant registers (ln2_recip, C23_73, ADJ_EXP)
   - For !APPROX: Initializes reciprocal lookup tables for negative input handling

2. **Per-tile execution** (`calculate_exponential`):
   - `copy_tile(c_0, 0, 0)` unpacks the tile from CB c_0 into DEST register 0
   - The SFPU operates on DEST in-place, processing 8 "faces" (32-element rows, processed as 16 pairs of even/odd columns)
   - For the LOADMACRO path: 16 LOADMACRO invocations (8 sanitize + 8 compute) process all 16 dest offsets
   - For the replay path: `lltt::replay` replays the recorded LM+SHFT2 pattern, then drains the pipeline
   - For the SFPI path: a loop over 8 iterations reads `dst_reg[0]`, computes the result, writes it back, advances
   - After SFPU completes, `pack_tile(0, c_2)` packs DEST[0] into CB c_2

3. **Synchronization**:
   - `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()` manage the dual-ownership protocol between math and pack RISC-Vs for DEST registers

#### SFPU Configuration

| Configuration | Value | Description |
|---------------|-------|-------------|
| `math_fidelity` | `MathFidelity::HiFi4` | Highest fidelity (not directly relevant for SFPU, applies to FPU matrix ops) |
| `math_approx_mode` | `false` (from `get_op_approx_mode`) | EXP returns false — but `fast_and_approx` is controlled by param0 |
| `fp32_dest_acc_en` | From operation params | When true, DEST uses float32 accumulation; selects `_sfpu_exp_f32_accurate_` path |
| `SFPU_OP_EXP_INCLUDE` | `1` (define) | Enables inclusion of `exp.h` via `sfpu_split_includes.h` |
| `SFPU_OP_CHAIN_0` | `exp_tile_init<P>(); exp_tile<P>(0);` | Macro-expanded SFPU operation chain |
| Template param0 | `fast_and_approximate_mode` (cast to uint32_t) | Controls `FAST_APPROX` template parameter (0 or 1) |

#### Hardware Compatibility Notes

- **Wormhole B0** and **Blackhole** share identical LLK-level implementations (in `tt_llk_wormhole_b0/` and `tt_llk_blackhole/`). Both files have the same `_calculate_exponential_`, `_init_exponential_`, and all helper functions.
- The arch-specific implementations (`hw/ckernels/{arch}/`) are also identical between Wormhole B0 and Blackhole, both providing `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, and `_sfpu_exp_f32_accurate_`.
- The SFPMAD instruction is compiled differently: on Wormhole it only supports `VD = VA * VB + VC`, while on Blackhole `SFFPMAD` supports `SFPMAD_MOD1_NEGATE_VA` and `SFPMAD_MOD1_NEGATE_VC`. The Cody-Waite implementation pre-negates constants to work on both.
- SFPLOADMACRO addr_mod parameter differs: Wormhole uses `ADDR_MOD_7` while Wormhole B0 uses `3` (the same register, different naming).
- The `_sfpu_exp_61f_` function (Moroz exp_61f, 6th degree polynomial) is defined but not used in the default paths. It exists as a higher-accuracy alternative between exp_21f and the full f32_accurate.

## Implementation Notes

1. **Operation chaining**: The `SFPU_OP_CHAIN_0` mechanism supports chaining multiple unary ops (e.g., exp then relu). For EXP standalone, the chain has one element.

2. **Parameterized vs non-parameterized**: EXP has a `fast_and_approximate_mode` parameter. When present, `get_op_init_and_func_parameterized` generates `exp_tile_init<P>(); exp_tile<P>(0);`. When absent (chaining without explicit params), `get_op_init_and_func_default` generates `exp_tile_init(); exp_tile(0);` using default template values (`approx=false`, `fast_and_approx=true`).

3. **Schraudolph algorithm**: The fast approximate path exploits the fact that reading an IEEE754 float as an integer is approximately linear in log2(value). Computing `i = A*x + (B-C)` and reinterpreting as float gives a fast exp approximation. The hardware pipeline (MAD → STOCH_RND → SHFT) computes this in ~2.5 cycles per element.

4. **Program caching**: `override_runtime_arguments` updates only buffer addresses on re-invocation, avoiding full program re-creation for same-shape tensors.

5. **Data type handling**: Input data type drives preprocessor defines (`INP_FLOAT32`, `INP_INT32`, `INP_UINT32`, `INP_FLOAT`) that may affect SFPU kernel behavior. For EXP, the primary paths are BFLOAT16 and FLOAT32.

6. **SubCoreGrid variant**: `UnarySubCoreGridProgramFactory` provides an alternative work distribution that uses caller-specified core subsets rather than the full compute grid. It divides tiles into blocks with `ntiles_per_block` tiles per block, rather than always using block_dim=1.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary element-wise operation (like exp) implemented in TTNN? What is the program factory structure for unary operations? What kernels (reader, compute, writer) are used and where are they located?"
   **Reason**: Needed to confirm the overall architecture and file layout of the unary operation framework
   **Key Findings**: Confirmed UnaryProgramFactory/UnarySubCoreGridProgramFactory structure, kernel paths, and the role of SFPU_OP_EXP_INCLUDE/SFPU_OP_CHAIN_0 defines

2. **Query**: "How do circular buffers work in tt-metal for SFPU unary operations? What is the typical CB configuration for a simple element-wise unary op like exp?"
   **Reason**: Needed to understand the CB pipeline pattern and synchronization protocol
   **Key Findings**: Confirmed double-buffering with 2 tiles, the init_sfpu → copy_tile → SFPU op → pack_tile flow, and tile_regs_acquire/commit/wait/release protocol

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to trace EXP-specific defines, kernel path selection, and parameter handling
   **Key Information**: EXP maps to `eltwise_sfpu.cpp` compute kernel, generates `SFPU_OP_EXP_INCLUDE=1` define, `math_approx_mode` returns false for EXP, `fast_and_approx` is param0

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Needed to understand the API-level dispatch from `exp_tile`/`exp_tile_init` to the SFPU kernel
   **Key Information**: `exp_tile` dispatches to `calculate_exponential` with template params for approx mode, fast_and_approx, fp32_dest_acc, scale, etc.

3. **Source**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Core SFPU kernel implementation shared between architectures
   **Key Information**: Contains `_sfpu_exp_`, `_calculate_exponential_`, `_init_exponential_` with multiple algorithm paths (LOADMACRO Schraudolph, replay-buffer Schraudolph, piecewise approx, SFPI precise)

4. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: Arch-specific implementations of improved exp algorithms
   **Key Information**: Contains `_sfpu_exp_21f_` (Moroz 2022), `_sfpu_exp_f32_accurate_` (Cody-Waite + Taylor), and the top-level `calculate_exponential` dispatch
