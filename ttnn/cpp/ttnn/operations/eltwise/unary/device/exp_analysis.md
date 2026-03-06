# EXP (Exponential) Implementation Analysis

## Overview

The EXP operation computes the element-wise exponential function `exp(x)` on each element of an input tensor. It is implemented as a **unary SFPU operation** using the shared `UnaryProgramFactory`, which is the common program factory for most unary elementwise operations. The EXP operation supports multiple precision modes: a fast approximate mode (default), a standard approximation mode, and a high-accuracy FP32 mode selected via `fp32_dest_acc_en`.

**Program Factory Path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | Tile (32x32 elements) |
| **Unit size** | 1 tile |
| **Total units** | `num_pages` = total tiles in the input tensor (`input.buffer()->num_pages()`) |
| **Loop structure** | Outer loop: `per_core_block_cnt` blocks. Inner loop: `per_core_block_dim` tiles per block (always 1 in this factory). Each tile is independently processed through copy->SFPU->pack. |

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Arbitrary (any rank) |
| **Dimension convention** | N/A (elementwise, shape-agnostic) |
| **Tensor layout** | TILE_LAYOUT (or ROW_MAJOR, detected at runtime) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32, INT32, or UINT32 |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or different if output dtype configured separately) |

### Layout Transformations

No explicit tilize/untilize or reshard operations. The program factory supports both TILE and ROW_MAJOR layouts transparently — when ROW_MAJOR, the CB page size is set to `buffer->page_size()` instead of `tile_size()`. The compute kernel still processes data as tiles regardless.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader | DRAM/L1 (src_buffer) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `cb_push_back(c_0, 1)` |
| 2 | Compute | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile(c_0, 0, 0)`, SFPU EXP, `pack_tile(0, c_2)`, `cb_pop_front(c_0, 1)`, `cb_reserve_back(c_2, per_core_block_dim)` / `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer | CB c_2 | DRAM/L1 (dst_buffer) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

**Note**: The compute kernel reserves/pushes the output CB at block granularity (`per_core_block_dim` tiles at a time), while the reader and writer operate on single tiles within their loops.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_1 | cb_tmp0 | Intermediate results (NOT used for EXP) | 2 tiles | 1 tile | Double | Compute | Compute | Program |
| c_2 | cb_output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

**Notes**:
- CB c_1 (tmp0) is only allocated for `HARDSHRINK`, `CBRT`, and `LOGIT` operations. It is **not created** for EXP.
- For BITCAST operations, the input CB uses the output data format instead of input data format.
- Double-buffering (capacity = 2 * block_size) allows overlap between producer and consumer.

## Pipeline Pattern Summary

All three active circular buffers (c_0, c_2) use double-buffering (capacity = 2 tiles, block size = 1 tile). This enables:
- Reader can fill one tile into c_0 while Compute processes the other
- Compute can fill one tile into c_2 while Writer drains the other
- Three-stage pipeline overlap: Reader, Compute, and Writer can all be active concurrently on different tiles

## Index Calculations

Index mapping uses `TensorAccessor` for both reader and writer:
- **Reader**: `TensorAccessorArgs(*src_buffer)` is passed as compile-time args. At runtime, pages are accessed sequentially from `start_id` to `start_id + num_pages`.
- **Writer**: `TensorAccessorArgs(*dst_buffer)` is passed as compile-time args. Same sequential pattern from `start_id`.
- **Page index mapping**: `noc_async_read_page(i, s, l1_write_addr)` and `noc_async_write_page(i, s, l1_read_addr)` use the TensorAccessor `s` to resolve logical page index `i` to a physical (bank_id, bank_offset) pair for interleaved buffers.

## Memory Access Patterns

### Read Pattern
- **Sequential**: Pages are read in ascending order from `start_id` to `end_id`
- **Single-page granularity**: One NoC read per tile
- **Blocking**: `noc_async_read_barrier()` is called after each page read (no batched reads)
- **Access type**: DRAM or L1 interleaved, via NoC0

### Write Pattern
- **Sequential**: Pages written in ascending order from `start_id` to `end_id`
- **Single-page granularity**: One NoC write per tile
- **Flushing**: `noc_async_writes_flushed()` after each page; `noc_async_write_barrier()` at the end
- **Access type**: DRAM or L1 interleaved, via NoC1

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major enumeration) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group split: group 1 gets `ceil(num_pages / num_cores)` tiles, group 2 gets `floor(num_pages / num_cores)` tiles |

**Core enumeration**: Cores are indexed as `core = {i / num_cores_y, i % num_cores_y}`, meaning column-major order (y varies fastest).

**Two compute kernels**: Separate kernel handles are created for core_group_1 and core_group_2 because they have different `per_core_block_cnt` compile-time arguments. The reader and writer kernels are shared across all cores (they receive per-core work counts as runtime args).

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0..N | TensorAccessorArgs | uint32_t[] | Serialized tensor accessor parameters for the source buffer (bank mapping, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for the output buffer (always `c_2` = 2) |
| 1..N | TensorAccessorArgs | uint32_t[] | Serialized tensor accessor parameters for the destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of blocks (tiles) this core processes |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for this factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) to read |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) to write |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Packed scalar parameter (unused for EXP; always 0) |
| 1 | packed_scalar2 | uint32_t | Packed scalar parameter (unused for EXP; always 0) |

**Note**: EXP is a parametrized unary op (its parameter is the `fast_and_approx` flag), but this parameter is encoded as a **template parameter** in the SFPU kernel defines, not as a runtime argument. The `packed_scalar1/2` runtime args remain 0 for EXP.

## Kernel Implementations

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_interleaved_start_id | BRISC (RISCV_0) | NOC0 | DRAM/L1 src_buffer | CB c_0 | Sequential page reads via TensorAccessor |
| eltwise_sfpu (compute) | TRISC (Math RISCV) | N/A | CB c_0 | CB c_2 | copy_tile, SFPU exp, pack_tile |
| writer_unary_interleaved_start_id | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 dst_buffer | Sequential page writes via TensorAccessor |

### Reader Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential loop from `start_id` to `start_id + num_pages`. For each page: reserve CB space, read page via NoC, barrier, push to CB. Supports optional `BACKWARDS` define for reverse iteration.

### Writer Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential loop from `start_id` to `start_id + num_pages`. For each page: wait for CB data, write page via NoC, flush, pop CB. Supports `OUT_SHARDED` define (wait for all pages at once) and `BACKWARDS` define.

### Compute Kernel
- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Key Logic**: The generic SFPU compute kernel used by most unary operations. The actual operation is injected via preprocessor defines (`SFPU_OP_CHAIN_0`). For EXP, this expands to `exp_tile_init<fast_and_approx>(); exp_tile<fast_and_approx>(0);`.

## SFPU Kernel Implementation

### SFPU Kernel File
- **API Header**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
- **Core Implementation (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Core Implementation (Wormhole B0)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
- **Legacy Implementation**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`

### SFPU Instructions and Intrinsics Used

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::exexp(val)` | Extract biased exponent from a float as an integer |
| `sfpi::exexp_nodebias(val)` | Extract exponent without bias removal (raw exponent bits) |
| `sfpi::exman8(val)` | Extract 8-bit mantissa with implicit leading bit (value in [1, 2)) |
| `sfpi::exman9(val)` | Extract 9-bit mantissa |
| `sfpi::shft(val, amt)` | Bitwise shift (left if positive, right if negative) |
| `sfpi::setexp(val, exp)` | Set the exponent field of a float to a specified value |
| `sfpi::addexp(val, delta)` | Add a constant to the exponent field (multiply by power of 2) |
| `sfpi::int32_to_float(val, 0)` | Convert integer to float |
| `sfpi::float_to_fp16b(val, 0)` | Convert float32 to bfloat16 with round-to-nearest-even |
| `sfpi::vec_min_max(a, b)` | Parallel min/max: after call, `a = min(a,b)` and `b = max(a,b)` |
| `sfpi::reinterpret<T>(val)` | Bitwise reinterpretation between vFloat/vInt/vUInt |
| `sfpi::s2vFloat16b(val)` | Convert scalar to vFloat in bfloat16 format |
| `sfpi::dst_reg[0]` | Read/write current DEST register position |
| `sfpi::dst_reg++` | Advance DEST register pointer to next row |
| `sfpi::vConst0`, `sfpi::vConst1` | Predefined vector constants (0.0f, 1.0f) |
| `sfpi::vConstFloatPrgm0/1/2` | Programmable constant registers (used in approximate mode init) |
| `PolynomialEvaluator::eval(...)` | Horner's method polynomial evaluation |
| `v_if / v_elseif / v_else / v_endif` | SFPU predicated execution (condition codes) |
| `v_and(cond)` | Narrow predication within a `v_if` block |

### SFPU Register Usage

- **DEST Registers (`dst_reg`)**: The SFPU reads input values from and writes results to the DEST register file. The compute kernel's `copy_tile` loads data from CB c_0 into DEST, and `pack_tile` reads from DEST to CB c_2. The SFPU operates on DEST in-place.
- **LRegs (Local Registers)**: SFPU local registers hold intermediate values during computation. All SFPI vector operations (`vFloat`, `vInt`) map to LReg operations. LRegs always operate in float32 internally, regardless of DEST accumulator format.
- **Programmable Constants (`vConstFloatPrgm0/1/2`, `vConstIntPrgm2`)**: Used in the approximate mode to store precomputed values (1/ln2, the 23_73 conversion constant, exponent adjustment). Set during `exp_init()` → `_init_exponential_()`.

### SFPU Execution Flow

The execution flow depends on the template parameters `APPROXIMATION_MODE` and `FAST_APPROX`, controlled by the `fast_and_approx` parameter from the user API.

#### Default Path: Non-Approximate Mode (`APPROXIMATION_MODE=false`)

This is the path taken when EXP is called without approximation (e.g., `ttnn.exp(tensor)`). The `calculate_exponential` function dispatches to `_sfpu_exp_improved_`, which selects between two algorithms based on `is_fp32_dest_acc_en`:

**When `fp32_dest_acc_en = false` (bfloat16 DEST) — uses `_sfpu_exp_21f_`:**

1. **Read from DEST**: `sfpi::vFloat val = sfpi::dst_reg[0]` — load one row (32 elements) from the current DEST position.
2. **Scale by 1/ln(2)**: `xlog2 = val * 1.4426950... + 127.0` — converts to base-2 and adds IEEE754 bias.
3. **Clamp**: `vec_min_max` clamps `xlog2` to [0, 255] to prevent overflow/underflow in intermediate calculations.
4. **Float-to-int conversion**: `_float_to_int32_for_exp21f_` extracts exponent and mantissa, then shifts mantissa by exponent to get the integer representation. This is a branch-free algorithm optimized for the constraint 0 <= val < 128.
5. **Split integer and fractional parts**: `exexp_nodebias` extracts the integer part (2^z_i), `exman9` extracts the fractional part (z_f).
6. **Polynomial approximation of 2^(z_f)**: A 2nd-degree polynomial `1.0017248 + 7.84e-8*x + 4.79e-15*x^2` approximates 2^x on [0, 2^23] (the fractional part is not yet normalized).
7. **Recombine**: `setexp(frac, exponential_part)` combines the integer exponent with the fractional mantissa to produce 2^(z_i) * 2^(z_f) = exp(val).
8. **BF16 rounding** (if `!is_fp32_dest_acc_en`): Explicit `float_to_fp16b` conversion with round-to-nearest-even to avoid truncation artifacts.
9. **Write back**: `sfpi::dst_reg[0] = result; sfpi::dst_reg++` — store result and advance to next row.
10. **Iteration**: Steps 1-9 repeat for `ITERATIONS` (default 8) rows, covering all 4 faces of the tile.

**When `fp32_dest_acc_en = true` — uses `_sfpu_exp_f32_accurate_`:**

1. **Read from DEST**: `sfpi::vFloat val = sfpi::dst_reg[0]`.
2. **Scale**: `z = val * INV_LN2` (1/ln(2) with full float32 precision).
3. **Special case handling** (predicated):
   - `z >= 128.0`: result = +infinity (overflow)
   - `z <= -127.0`: result = 0.0 (underflow)
   - `exponent == 255`: result = NaN
4. **Round to nearest integer**: `_sfpu_round_nearest_int32_` uses the Hacker's Delight trick (add 2^23 + 2^22, subtract back) for round-to-nearest-even, producing both float `k` and int `k_int`.
5. **Cody-Waite range reduction**: `r = val - k*LN2_HI - k*LN2_LO` using split ln(2) constants for extended precision. The constants are pre-negated to enable SFPMAD optimization (`k * (-LN2_HI) + val`).
6. **7th-order Taylor series**: `PolynomialEvaluator::eval(r, 1, 1, 1/2!, 1/3!, 1/4!, 1/5!, 1/6!, 1/7!)` computes exp(r) with < 1 ULP accuracy.
7. **Scale by 2^k**: Extract current exponent of polynomial result, add `k_int`, set new exponent. This is equivalent to `ldexp(p, k)`.
8. **Write back**: Result stored to DEST, advance pointer.

#### Approximate Path (`APPROXIMATION_MODE=true, FAST_APPROX=true`)

When the user passes `fast_and_approx=true` (the default template parameter for EXP via the op_chain):

1. **Init** (`_init_exponential_`): Pre-loads constants into programmable registers:
   - `vConstFloatPrgm0`: 1/ln(2) in FP16b
   - `vConstFloatPrgm1`: The 23_73 constant for FxP conversion
   - `vConstFloatPrgm2`: Exponent adjustment
   - If `CLAMP_NEGATIVE=true`: additionally loads -88.5 threshold for input clamping.

2. **Execution** (`_calculate_exponential_`): When `FAST_APPROX && CLAMP_NEGATIVE`:
   - First pass: Sanitize inputs by clamping values below -88.5 to -88.5 (prevents incorrect outputs from very negative inputs).
   - Second pass: Apply the fast approximation via `_calculate_exponential_approx_`:
     - Multiply by 1/ln(2), convert to 7.3 fixed-point format
     - Shift and reinterpret as float — this directly constructs the IEEE754 representation of 2^x
   - This is based on Schraudolph's fast exp algorithm.

### SFPU Configuration

| Configuration | Value | Description |
|---------------|-------|-------------|
| **Math Fidelity** | `MathFidelity::HiFi4` | Highest fidelity for FPU operations (used by unpack/pack, not directly by SFPU) |
| **Math Approx Mode** | `false` (for EXP) | `get_op_approx_mode` returns `false` for all ops (default case). This sets the global `MATH_FIDELITY` register. |
| **fp32_dest_acc_en** | User-configurable | When true, DEST uses float32 accumulation; selects `_sfpu_exp_f32_accurate_` |
| **preserve_fp32_precision** | User-configurable | When true, sets `UnpackToDestFp32` mode for CB c_0 and c_1 |
| **bfp8_pack_precise** | User-configurable | Enables precise BFP8 packing |
| **SFPU_OP_EXP_INCLUDE** | `"1"` | Preprocessor define that triggers `#include "api/compute/eltwise_unary/exp.h"` |
| **SFPU_OP_CHAIN_0** | `"exp_tile_init<P>(); exp_tile<P>(0);"` | The actual operation chain injected into the compute kernel, where P is the fast_and_approx flag |
| **INP_FLOAT32/INP_FLOAT/INP_INT32/INP_UINT32** | Based on input dtype | Preprocessor define indicating input data format |

### Hardware Compatibility Notes

- **Wormhole B0 and Blackhole**: Both architectures use identical high-level `ckernel_sfpu_exp.h` files in `tt_metal/hw/ckernels/`. The implementations of `_sfpu_exp_21f_`, `_sfpu_exp_61f_`, and `_sfpu_exp_f32_accurate_` are the same across both.
- **Legacy LLK layer** (`tt_metal/third_party/tt_llk/`): Contains the approximate mode implementations (`_calculate_exponential_`, `_init_exponential_`, `_sfpu_exp_`) which differ between architectures primarily in register programming details but use the same algorithmic approach.
- **SFPMAD instruction**: The Cody-Waite range reduction in `_sfpu_exp_f32_accurate_` explicitly pre-negates constants to enable single-instruction `VD = VA * VB + VC` on both Wormhole (SFPMAD) and Blackhole (SFFPMAD, which also supports negate modifiers).
- **Predicated execution**: Both architectures support `v_if/v_else/v_endif` via SFPU condition codes. The approximate fast path uses `v_and` for narrowing predication in the squaring loop.

## Implementation Notes

1. **EXP is a parametrized type**: The `is_parametrized_type` function returns `true` for `UnaryOpType::EXP`. Its parameter (index 0) is the `fast_and_approx` boolean flag, cast to float. When parametrized, the defines become `exp_tile_init<{param0}u>()` and `exp_tile<{param0}u>(0)`, where param0 is 0 or 1.

2. **Non-parametrized fallback**: When EXP is used without parameters (line 615 of `unary_op_utils.cpp`), the defines are simply `exp_tile_init()` and `exp_tile(0)`, using default template arguments (`approx=false, fast_and_approx=true`).

3. **Three algorithmic variants**:
   - `_sfpu_exp_21f_<false>`: Moroz et al. 2nd-degree polynomial, bfloat16 DEST (default non-approx path)
   - `_sfpu_exp_f32_accurate_`: Cody-Waite + 7th-order Taylor, float32 DEST (high-accuracy path)
   - `_calculate_exponential_` (approx): Schraudolph's fast exp with optional input clamping

4. **`_sfpu_exp_61f_`**: A 6th-degree polynomial variant (Moroz exp_61f algorithm) exists in the codebase but is not directly called by the default `calculate_exponential` dispatch. It uses `addexp(frac, -23)` to normalize the fractional part before polynomial evaluation, yielding higher accuracy than `_sfpu_exp_21f_` but lower than the full Cody-Waite approach.

5. **VectorMode::RC**: The `exp_tile` function defaults to `VectorMode::RC`, meaning it processes all 4 faces of a tile (each face = 16x16 elements). The LLK dispatch (`_llk_math_eltwise_unary_sfpu_params_`) iterates over all 4 faces, calling the SFPU function once per face (8 iterations per face = 8 rows of 32 elements = 256 elements per face, 1024 per tile).

6. **Program caching**: The factory returns shared variables (`unary_reader_kernel_id`, `unary_writer_kernel_id`, `num_cores`, `num_cores_y`) for `override_runtime_arguments`, which only updates buffer addresses on subsequent calls without recreating the program.

7. **Two-group kernel split**: Because `per_core_block_cnt` is a compile-time argument, two separate compute kernel instances are created when work doesn't divide evenly. Core group 1 gets `ceil(N/C)` tiles and core group 2 gets `floor(N/C)` tiles, where N is total tiles and C is total cores.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary elementwise operation program factory implemented? What kernels does it use?"
   **Reason**: Needed to understand the overall architecture and factory selection logic before diving into code.
   **Key Findings**: Confirmed three factories (Unary, UnarySharded, UnarySubCoreGrid), identified kernel file paths, learned about the define-based operation injection mechanism.

2. **Query**: "What is the SFPU EXP operation in TTNN? How is it invoked in the compute kernel? What LLK APIs are used?"
   **Reason**: Needed to understand the SFPU-level implementation and the chain from `exp_tile` to the actual computation functions.
   **Key Findings**: Identified `calculate_exponential` as the core dispatch function, learned about the three implementation variants (_sfpu_exp_21f_, _sfpu_exp_f32_accurate_, _calculate_exponential_ for approx mode), discovered the SFPI intrinsics used.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Needed to understand how EXP maps to its compute kernel path, defines, and parameter handling.
   **Key Information**: EXP falls through to `default` case returning `"eltwise_sfpu.cpp"`. The `get_block_defines` function generates `SFPU_OP_EXP_INCLUDE` and `SFPU_OP_CHAIN_0` defines. `get_op_approx_mode` returns `false` for all ops by default.

2. **Source**: `tt_metal/hw/inc/api/compute/eltwise_unary/exp.h`
   **Reason**: Primary API header defining `exp_tile_init` and `exp_tile` with all template parameters.
   **Key Information**: Documented all template parameters (approx, fast_and_approx, scale_en, skip_positive_check, input_clamping, iterations) and the macro dispatch to `calculate_exponential`.

3. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
   **Reason**: Core SFPU implementation containing all three exp algorithm variants.
   **Key Information**: Full algorithmic details of `_sfpu_exp_21f_`, `_sfpu_exp_f32_accurate_`, and `_sfpu_exp_61f_` including mathematical derivations, polynomial coefficients, and special case handling.

4. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`
   **Reason**: Contains the approximate mode implementation and initialization.
   **Key Information**: The `_calculate_exponential_` and `_init_exponential_` functions implementing Schraudolph's fast exp algorithm and the programmable constant register setup.

5. **Source**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
   **Reason**: Needed to understand how `SFPU_TEMPLATE_INIT_KERNEL` and `SFPU_TEMPLATE_PARAMS_KERNEL_FN` macros expand.
   **Key Information**: `SFPU_TEMPLATE_PARAMS_KERNEL_FN` calls `_llk_math_eltwise_unary_sfpu_params_` with all template args forwarded to `calculate_exponential`.

6. **Source**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Needed to understand the LLK dispatch mechanism for SFPU operations.
   **Key Information**: `_llk_math_eltwise_unary_sfpu_params_` handles VectorMode dispatch (R, C, RC) by iterating over the appropriate faces and calling the SFPU function for each face.
