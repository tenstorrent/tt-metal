# ERFINV Implementation Analysis

## Overview

ERFINV computes the element-wise inverse error function on each element of the input tensor. Given an input `x` in the range `(-1, 1)`, it returns the value `y` such that `erf(y) = x`. The function is undefined for `|x| > 1` (returns NaN) and returns +/-infinity at `|x| = 1`.

ERFINV is a standard unary SFPU operation that uses the shared unary program factory. It does not have a dedicated program factory or specialized kernels -- it plugs into the generic `eltwise_sfpu.cpp` compute kernel via preprocessor defines.

**Program factory path**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.cpp`

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile |
| **Unit size** | 1 tile (32x32 elements) |
| **Total units** | `num_pages` = total number of tiles in the input tensor |
| **Loop structure** | Outer loop over `per_core_block_cnt` blocks, inner loop over `per_core_block_dim` tiles (always 1 for this factory) |

## Tensor Format and Layout

### Input Tensor

| Property | Input Tensor |
|----------|--------------|
| **Logical shape** | Any shape (flattened to tiles) |
| **Dimension convention** | N/A (operates element-wise on all tiles) |
| **Tensor layout** | TILE_LAYOUT (also supports ROW_MAJOR) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | BFLOAT16, FLOAT32 |

### Output Tensor

| Property | Output Tensor |
|----------|---------------|
| **Logical shape** | Same as input |
| **Dimension convention** | Same as input |
| **Tensor layout** | Same as input |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM or L1 |
| **Data type** | Same as input (or may differ if output dtype is configured) |

### Layout Transformations

No layout transformations are performed. The operation is a pure element-wise unary; input and output have identical shapes and layouts. The data format conversion between input and output (if different) is handled transparently by the unpacker and packer hardware.

## Data Flow Pattern

| Stage | Kernel | Reads From | Writes To | CB Operations |
|-------|--------|------------|-----------|---------------|
| 1 | Reader (BRISC) | DRAM/L1 (interleaved pages) | CB c_0 | `cb_reserve_back(c_0, 1)`, `noc_async_read_page`, `noc_async_read_barrier`, `cb_push_back(c_0, 1)` |
| 2 | Compute (TRISC) | CB c_0 | CB c_2 | `cb_wait_front(c_0, 1)`, `copy_tile`, `erfinv_tile_init` + `erfinv_tile`, `pack_tile`, `cb_pop_front(c_0, 1)`, `cb_reserve_back(c_2, per_core_block_dim)` / `cb_push_back(c_2, per_core_block_dim)` |
| 3 | Writer (NCRISC) | CB c_2 | DRAM/L1 (interleaved pages) | `cb_wait_front(c_2, 1)`, `noc_async_write_page`, `cb_pop_front(c_2, 1)` |

### Detailed Data Flow

1. **Reader** iterates over its assigned tile range (`start_id` to `start_id + num_pages`). For each tile, it reserves one page in CB c_0, issues a NoC read from the interleaved buffer using `TensorAccessor` for address translation, waits for the read to complete, then pushes one page.

2. **Compute** runs an outer loop over `per_core_block_cnt` blocks. For each block, it reserves `per_core_block_dim` (=1) output pages in CB c_2, then for each tile: acquires tile registers, waits for one input tile in CB c_0, copies it to DST register 0, executes the SFPU op chain (`erfinv_tile_init(); erfinv_tile(0);`), commits tile registers, waits for pack, packs the result from DST register 0 to CB c_2, pops the input tile from CB c_0, and releases tile registers. After the inner loop, pushes the block to CB c_2.

3. **Writer** iterates over the same tile range. For each tile, it waits for one page in CB c_2, reads the L1 address, issues a NoC write to the interleaved output buffer, flushes writes, and pops one page. After all tiles, issues a write barrier.

## Circular Buffer Configuration

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | src0 | Input staging | 2 tiles | 1 tile | Double | Reader | Compute | Program |
| c_2 | output | Output staging | 2 tiles | 1 tile | Double | Compute | Writer | Program |

Note: CB c_1 (tmp0) is **not** allocated for ERFINV. It is only created for HARDSHRINK, CBRT, and LOGIT operations.

### CB Page Size

- For TILE_LAYOUT: page size = `tile_size(cb_data_format)` which is format-dependent (e.g., 2048 bytes for BFLOAT16 with 32x32 tiles, 4096 bytes for FLOAT32).
- For ROW_MAJOR: page size = `buffer->page_size()`.

## Pipeline Pattern Summary

Both CB c_0 and CB c_2 have capacity = 2 tiles and block size = 1 tile, yielding **double-buffered** pipelines. This allows the reader to fill one slot while compute processes the other, and similarly compute can fill one output slot while the writer drains the other. This enables overlap between all three pipeline stages.

## Index Calculations

Index-to-address mapping is handled entirely by `TensorAccessor`, which is constructed from `TensorAccessorArgs` at compile time. The key parameters are packed into compile-time args via `TensorAccessorArgs(*buffer).append_to(compile_time_args)`.

On the device side, the reader constructs:
```cpp
const auto s = TensorAccessor(src_args, src_addr, page_bytes);
```

The `noc_async_read_page(i, s, l1_write_addr)` call uses the TensorAccessor to translate logical page index `i` to a physical {NoC address, bank offset} pair. For interleaved buffers, this involves:
- Computing `bank_id = page_index % num_banks`
- Computing `page_offset = page_index / num_banks * page_size`
- Looking up the bank's NoC coordinates and base address

The writer uses an analogous `TensorAccessor` constructed from `dst_args` at compile-time arg index 1.

## Memory Access Patterns

### Read Pattern

**Sequential page reads**: The reader iterates linearly from `start_id` to `start_id + num_pages`, reading one page at a time. For interleaved buffers, consecutive page indices map to different DRAM banks in a round-robin pattern, distributing NoC traffic across banks. Each read is immediately followed by a barrier (`noc_async_read_barrier`), making this a synchronous one-page-at-a-time pattern.

### Write Pattern

**Sequential page writes**: The writer iterates linearly over the same page range, writing one page at a time. Writes use `noc_async_writes_flushed()` after each page (not a full barrier), which only ensures the write has left the local NoC interface. A full `noc_async_write_barrier()` is issued after all pages are written.

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 2D (column-major iteration) |
| **Grid dimensions** | `compute_with_storage_grid_size` (device-dependent, e.g., 8x8 on Wormhole) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_pages_per_core_group_1` or `num_pages_per_core_group_2` tiles |
| **Load balancing** | Two-group remainder distribution |

### Remainder Handling

`split_work_to_cores(grid_size, num_pages)` divides tiles across cores:
- If `num_pages` divides evenly: all cores get `num_pages / num_cores` tiles (core_group_2 is empty).
- If remainder exists: `core_group_1` gets `floor(num_pages / num_cores) + 1` tiles, `core_group_2` gets `floor(num_pages / num_cores)` tiles. The number of cores in group_1 equals the remainder.

### Core Enumeration

Cores are enumerated in **column-major** order: `core = {i / num_cores_y, i % num_cores_y}`. This means core index 0 maps to (0,0), index 1 to (0,1), etc., filling columns first.

Two separate compute kernels are compiled -- one for core_group_1 and one for core_group_2 -- because `per_core_block_cnt` is a compile-time argument and differs between the groups.

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0+ | TensorAccessorArgs(src) | uint32_t[] | Packed tensor accessor parameters for source buffer (rank, num_banks, shape, bank coords) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | cb_id_out | uint32_t | Output circular buffer index (= c_2 = 2) |
| 1+ | TensorAccessorArgs(dst) | uint32_t[] | Packed tensor accessor parameters for destination buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | per_core_block_cnt | uint32_t | Number of tile blocks to process on this core |
| 1 | per_core_block_dim | uint32_t | Tiles per block (always 1 for this factory) |

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) to read on this core |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer base address in DRAM/L1 |
| 1 | num_pages | uint32_t | Number of pages (tiles) to write on this core |
| 2 | start_id | uint32_t | Starting page index for this core |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | packed_scalar1 | uint32_t | Unused for ERFINV (always 0) |
| 1 | packed_scalar2 | uint32_t | Unused for ERFINV (always 0) |

### Preprocessor Defines (Compile-Time)

The program factory sets these defines on the compute kernel:

| Define | Value | Description |
|--------|-------|-------------|
| `SFPU_OP_ERFINV_INCLUDE` | `1` | Enables inclusion of `erfinv.h` via `sfpu_split_includes.h` |
| `SFPU_OP_CHAIN_0` | `erfinv_tile_init(); erfinv_tile(0);` | The SFPU op chain executed per tile |
| `SFPU_OP_CHAIN_0_INIT_0` | `erfinv_tile_init();` | Initialization call |
| `SFPU_OP_CHAIN_0_FUNC_0` | `erfinv_tile(0);` | Per-tile function call |
| `INP_FLOAT` or `INP_FLOAT32` | `1` | Input data type indicator |

### Compute Configuration

| Parameter | Value |
|-----------|-------|
| `math_fidelity` | HiFi4 |
| `fp32_dest_acc_en` | From `args.fp32_dest_acc_en` |
| `math_approx_mode` | `false` (ERFINV returns false from `get_op_approx_mode`) |

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_interleaved_start_id | BRISC (RISCV_0) | NOC0 | DRAM/L1 interleaved buffer | CB c_0 | Sequential page read via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Key Logic**: Reads `num_pages` tiles starting from `start_id`. Each iteration: reserves 1 page in CB c_0, computes L1 write address, issues `noc_async_read_page` using TensorAccessor, barriers, then pushes. Supports optional `BACKWARDS` define for reverse iteration (not used in standard ERFINV).

### Compute Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| eltwise_sfpu | TRISC (math RISCV) | N/A | CB c_0 | CB c_2 | SFPU erfinv computation |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **Key Logic**: Generic SFPU compute kernel. Calls `init_sfpu(c_0, c_2)` to configure unpacker/packer. Outer loop runs `per_core_block_cnt` times, inner loop runs `per_core_block_dim` (=1) times. Per tile: acquires DST registers, waits for input in c_0, copies tile to DST[0], executes the `SFPU_OP_CHAIN_0` macro (which expands to `erfinv_tile_init(); erfinv_tile(0);`), commits and waits for packing, packs the result from DST register 0 to CB c_2, pops input, releases registers. Pushes block after inner loop.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | NCRISC (RISCV_1) | NOC1 | CB c_2 | DRAM/L1 interleaved buffer | Sequential page write via TensorAccessor |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Writes `num_pages` tiles starting from `start_id`. Each iteration: waits for 1 page in CB c_2, reads L1 address, issues `noc_async_write_page` using TensorAccessor, flushes, pops. Final `noc_async_write_barrier` after loop. Supports `OUT_SHARDED` define (not used in interleaved factory).

### SFPU Implementation

- **File (Wormhole)**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`
- **File (Blackhole)**: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`
- **Algorithm**: Based on the Winitzki (2008) approximation: `erfinv(x) = sign(x) * sqrt(-2/(pi*a) - log(1-x^2)/2 + sqrt((2/(pi*a) + log(1-x^2)/2)^2 - log(1-x^2)/a))` where `a = 0.147`.
- **Key constants**: `TwoPiA = 4.330746750799873` (2/(pi*a)), `OneDivA = 6.802721088435375` (1/a).
- **SFPU iteration count**: 8 iterations per call (processes 8 elements of the 32-wide SFPU vector, requiring 4 calls to cover a full 32-element row; the tile has 32 rows, but the SFPU processes face elements).
- **Dependencies**: Calls `calculate_log_body` (from `ckernel_sfpu_log.h`) for natural log computation, and a custom `calculate_sqrt_custom` using the fast inverse square root trick (magic number `0x5f37`).
- **Edge cases**: `|x| == 1` returns +/-infinity; `|x| > 1` returns NaN; the sign of the result is restored from the original input using `sfpi::setsgn`.
- **Initialization**: `erfinv_init()` calls `log_init<false, false, false>()` since the algorithm depends on the log SFPU function.

## Implementation Notes

1. **No scalar parameters**: Unlike operations such as HARDSHRINK or WHERE_TSS, ERFINV takes no scalar parameters. The `packed_scalar1` and `packed_scalar2` runtime args passed to the compute kernel are always 0.

2. **Shared program factory**: ERFINV does not have its own program factory. It uses the general-purpose `UnaryProgramFactory` and is differentiated entirely through preprocessor defines set by `get_block_defines` and `update_macro_defines`.

3. **No temporary CB needed**: CB c_1 is only allocated for HARDSHRINK, CBRT, and LOGIT. ERFINV does not require intermediate scratch space at the CB level.

4. **Identical Wormhole/Blackhole implementations**: The SFPU kernel code is byte-for-byte identical between the two architecture backends, meaning the algorithm has no architecture-specific optimizations.

5. **Custom sqrt implementation**: The SFPU erfinv kernel uses its own `calculate_sqrt_custom` function instead of the standard SFPU sqrt. This uses the fast inverse square root algorithm (Quake III-style) with two Newton-Raphson refinement iterations, multiplied by the original value to get the square root.

6. **FP32 intermediate log**: The `calculate_log_body` call uses the template parameter `true` for the third parameter (`use_fp32`), ensuring intermediate log computation avoids bfloat16 rounding errors that would accumulate in the multi-step erfinv formula.

7. **Program caching**: The `override_runtime_arguments` method only updates buffer addresses (indices 0 of reader/writer runtime args). This enables efficient program reuse when only tensor addresses change between invocations.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the unary program factory structured for eltwise unary operations?"
   **Reason**: Needed to understand the three factory variants (interleaved, sharded, sub-core-grid) and how kernels are configured.
   **Key Findings**: Three variants exist; ERFINV uses the default interleaved variant. The factory creates reader/compute/writer kernels with TensorAccessor-based address translation.

2. **Query**: "How does split_work_to_cores work?"
   **Reason**: Needed to understand core distribution and remainder handling.
   **Key Findings**: Returns two core groups: group_1 gets `floor + 1` tiles (remainder cores), group_2 gets `floor` tiles. Column-major core enumeration.

3. **Query**: "How does TensorAccessorArgs work in tt-metal?"
   **Reason**: Needed to understand compile-time vs runtime argument generation for reader/writer kernels.
   **Key Findings**: TensorAccessorArgs packs tensor layout info (rank, num_banks, shape, bank coords) into compile-time args. On-device, TensorAccessor translates logical page indices to physical NoC addresses.

### Documentation References

1. **Source**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Traced ERFINV through `get_macro_definition` (returns `SFPU_OP_ERFINV_INCLUDE`), `get_block_defines` (generates SFPU_OP_CHAIN defines), `get_compute_kernel_path` (falls through to default `eltwise_sfpu.cpp`), and `get_op_approx_mode` (returns false).
   **Key Information**: ERFINV has no special-case handling in the program factory -- it uses all default paths.

2. **Source**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h`
   **Reason**: Contains the actual SFPU algorithm implementation.
   **Key Information**: Uses Winitzki's 2008 approximation with custom sqrt and log dependencies. Processes 8 SFPU elements per iteration. Handles edge cases (|x|=1 gives inf, |x|>1 gives NaN).

## SFPU Kernel Implementation

This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/erfinv.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro layer) and `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (params dispatch) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (combined with LLK dispatch) |

### Call Chain

1. The compute kernel macro `SFPU_OP_CHAIN_0` expands to `erfinv_tile_init(); erfinv_tile(0);`, calling the API-level functions defined in `erfinv.h`.
2. `erfinv_tile(idst)` invokes the macro `SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_erfinv, RC, APPROX, idst)`, which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_erfinv<APPROXIMATE>, idst, (int)VectorMode::RC)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DST write address, stalls until the SFPU is ready, then iterates over 4 faces (VectorMode::RC), calling `ckernel::sfpu::calculate_erfinv<APPROXIMATE>()` for each face and advancing the DEST pointer by 16 rows (2x `TTI_SETRWC` with increment 8) between faces.
4. `erfinv_tile_init()` invokes `SFPU_INIT_KERNEL_CALL(erfinv, sfpu::erfinv_init, APPROX)`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::erfinv, APPROXIMATE>(erfinv_init<APPROXIMATE>)`. This first calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::erfinv>()` to configure the SFPU config register, set up ADDR_MOD_7 (all increments = 0), and reset counters, then calls `erfinv_init<APPROXIMATE>()` which programs the LUT registers (`vConstFloatPrgm0`, `vConstFloatPrgm1`, `vConstFloatPrgm2`) with log-related constants.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_sqrt_custom(sfpi::vFloat in) { // APPROXIMATION_MODE is unused in this function
    sfpi::vFloat val = in;
    sfpi::vFloat out;
    v_if(val != 0.0f) {
        // Fast inverse square root (Quake III algorithm) with magic constant 0x5f37
        sfpi::vUInt magic = sfpi::reinterpret<sfpi::vUInt>(sfpi::vFloat(sfpi::s2vFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx; // Newton-Raphson iteration 1
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx; // Newton-Raphson iteration 2
        out = approx * val; // Convert 1/sqrt(x) to sqrt(x) by multiplying by x
    }
    v_else { out = val; } // sqrt(0) = 0
    v_endif;
    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in) {
    // Implementation notes, see the original file for more details

    // Compute log(1 - x^2)
    sfpi::vFloat log_value = in * in;
    log_value = 1 - log_value;
    log_value = calculate_log_body<false, false, true>(log_value, 0); // FAST_APPROX=false, HAS_BASE_SCALING=false, is_fp32_dest_acc_en=true

    sfpi::vFloat temp = log_value * 0.5;

    // a = 0.147; TwoPiA = 2/(pi*a) = 4.330746750799873; OneDivA = 1/a = 6.802721088435375
    constexpr float TwoPiA = 4.330746750799873f;
    constexpr float OneDivA = 6.802721088435375f;

    // tmp = -( 2/(pi*a) + log(1 - x^2)/2 )
    temp = TwoPiA + temp;
    temp = -temp;

    // calculated_value = temp + sqrt( temp^2 - log_value / a )
    sfpi::vFloat calculated_value = (temp * temp) - (log_value * OneDivA);
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(calculated_value);
    calculated_value = temp + intermediate_result;

    // result = sqrt(calculated_value)
    sfpi::vFloat result = calculate_sqrt_custom<false>(calculated_value);

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_erfinv() { // APPROXIMATION_MODE=false (get_op_approx_mode returns false for erfinv)
    constexpr int ITERATIONS = 8;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0]; // Load element from DEST register
        sfpi::vFloat result;

        // erfinv(-x) = -erfinv(x), so compute on |x| and restore sign later
        sfpi::vFloat abs_v = sfpi::abs(v);

        v_if(abs_v == 1.0f) { result = std::numeric_limits<float>::infinity(); } // erfinv(+/-1) = +/-inf
        v_elseif(abs_v > 1.0f) {
            result = std::numeric_limits<float>::quiet_NaN(); // undefined for |x| > 1
        }
        v_else { result = calculate_erfinv_body<true>(abs_v); }
        v_endif;

        result = sfpi::setsgn(result, v); // Restore original sign from input

        sfpi::dst_reg[0] = result; // Store result back to DEST register
        sfpi::dst_reg++; // Advance DEST register pointer by 1 row
    }
}

template <bool APPROXIMATION_MODE>
void erfinv_init() {
    log_init<false, false, false>(); // Programs vConstFloatPrgm0=ln(2), vConstFloatPrgm1, vConstFloatPrgm2 for log polynomial
}
```

The `calculate_log_body` function (from `ckernel_sfpu_log.h`) called with `<false, false, true>` (FAST_APPROX=false, HAS_BASE_SCALING=false, is_fp32_dest_acc_en=true) computes `ln(x)` using:

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_log.h
// (relevant excerpt for the is_fp32_dest_acc_en=true path used by erfinv)

template <bool HAS_BASE_SCALING>
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat val, const uint log_base_scale_factor) {
    sfpi::vFloat result;

    sfpi::vInt exp = sfpi::exexp(val); // Extract debiased exponent

    v_if(sfpi::reinterpret<sfpi::vInt>(val) == 0x7F800000) {
        result = std::numeric_limits<float>::infinity();
    }
    v_elseif(exp == 128 || val < 0.f) {
        result = std::numeric_limits<float>::quiet_NaN();
    }
    v_elseif(val == 0.0f) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_else {
        sfpi::vFloat m = sfpi::setexp(val, 127); // Normalize mantissa to [1, 2)

        constexpr float SQRT2 = 1.4142135381698608f;
        v_if(m >= SQRT2) {
            m = m * 0.5f;
            exp = exp + 1;
        }
        v_endif;

        // z = (m - 1) / (m + 1), maps to [-0.172, 0.172]
        sfpi::vFloat m_minus_1 = m - sfpi::vConst1;
        sfpi::vFloat m_plus_1 = m + sfpi::vConst1;
        sfpi::vFloat m_plus_1_recip = _sfpu_reciprocal_<2>(m_plus_1); // 2 Newton-Raphson iterations
        sfpi::vFloat z = m_minus_1 * m_plus_1_recip;
        sfpi::vFloat z2 = z * z;

        // Horner's method: p = 1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11
        sfpi::vFloat p = PolynomialEvaluator::eval(
            z2,
            sfpi::vConst1,
            0.3333333333333333f,
            0.2f,
            0.14285714285714285f,
            0.1111111111111111f,
            .09090909090909091f);

        sfpi::vFloat ln_m = 2.0f * (z * p);

        v_if(exp < 0) {
            sfpi::vInt exp_abs = ~exp + 1; // Two's complement negation
            exp = sfpi::setsgn(exp_abs, 1); // Convert to sign-magnitude for int32_to_float
        }
        v_endif;

        sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);

        constexpr float LN2 = 0.69314718246459961f;
        result = expf * LN2 + ln_m; // ln(x) = exp * ln(2) + ln(m)
    }
    v_endif;

    return result;
}
```

### SFPU Instructions Used

The erfinv kernel and its dependencies use the following SFPU instructions and intrinsics:

| Instruction/Intrinsic | Description |
|----------------------|-------------|
| `sfpi::dst_reg[0]` (SFPLOAD/SFPSTORE) | Loads from / stores to the DEST register file. Each access reads or writes one element from the current DEST row. |
| `sfpi::dst_reg++` (SFPINCRWC) | Increments the DEST register row pointer by 1, advancing to the next element in the SFPU iteration. |
| `sfpi::abs(v)` | Computes absolute value by clearing the sign bit of a float. |
| `sfpi::setsgn(result, v)` (SFPSETSGN) | Copies the sign bit from `v` to `result`, used to restore the original sign after computing on `|x|`. |
| `sfpi::setexp(val, 127)` (SFPSETEXP) | Sets the exponent field of a float to 127 (bias), normalizing the mantissa to the range [1, 2). Used in log computation. |
| `sfpi::exexp(val)` (SFPEXEXP) | Extracts the debiased exponent from a float as an integer. Returns `exp - 127` for normal floats. |
| `sfpi::int32_to_float(exp, 0)` (SFPCAST) | Converts a sign-magnitude integer to a floating-point value. The second argument (0) specifies no shift. |
| `sfpi::reinterpret<vUInt>(vFloat)` (SFPLUT) | Reinterprets the bit pattern of a float as an unsigned integer (or vice versa) without conversion. Used for bit manipulation in the fast inverse sqrt. |
| `sfpi::s2vFloat16b(0x5f37)` | Creates a BFloat16 constant from a raw hex value and broadcasts it across the SFPU vector lanes. |
| `sfpi::float_to_fp16b(result, 0)` (SFPCAST) | Converts a float32 value to BFloat16 format. Not used in erfinv's fp32 path but present in log's non-fp32 path. |
| `v_if / v_elseif / v_else / v_endif` (SFPSETCC/SFPENCC/SFPCOMPC) | Conditional execution macros that set/enable/complement the SFPU condition codes (CC). These control per-lane predication: only lanes with CC=true execute subsequent instructions. |
| `_sfpu_reciprocal_<2>(x)` | Computes 1/x using the SFPU's built-in reciprocal approximation with 2 Newton-Raphson refinement iterations. Used inside `calculate_log_f32_body`. |
| `PolynomialEvaluator::eval(...)` | Evaluates a polynomial using Horner's method. Compiles down to a sequence of SFPMAD (multiply-add) instructions. |
| `TTI_STALLWAIT` | Stalls the pipeline until the specified condition is met. Used at SFPU entry (wait for math unit) and exit (wait for SFPU completion). |
| `TTI_SETRWC` | Sets the read/write counters for DEST register addressing. Used by the params dispatch to advance between tile faces. |
| Arithmetic operators (`*`, `+`, `-`, `>>`) | Map to SFPMUL, SFPADD, SFPSUB, and SFPSHFT instructions respectively. These are the core ALU operations performed on SFPU vector registers. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST registers (dst_reg)** | The primary data source and sink. Each iteration loads one element from DEST row 0, processes it through the erfinv algorithm, writes the result back to DEST row 0, then increments the row pointer. Over 8 iterations, this processes 8 consecutive rows of a 16-row face. |
| **SFPU LREGs (L0-L7)** | Implicit temporaries used by the SFPI compiler for intermediate values (`val`, `approx`, `log_value`, `temp`, `calculated_value`, `result`, etc.). The compiler allocates these automatically. The erfinv kernel has high register pressure due to nested function calls (erfinv_body calls log_body and sqrt_custom). |
| **vConstFloatPrgm0** | Programmed to `0.69314718246459961f` (ln(2)) during `erfinv_init()` via `log_init`. Used by `calculate_log_f32_body` for the exponent correction term `exp * ln(2)`. |
| **vConstFloatPrgm1** | Programmed to `-2.0069785118103027` during `log_init`. Used by the non-fp32 `calculate_log_body` polynomial (not the fp32 path used by erfinv). |
| **vConstFloatPrgm2** | Programmed to `3.767500400543213` during `log_init`. Same as above -- used by the non-fp32 log path. |
| **vConst1** | Hardware constant register holding `1.0f`. Used in `calculate_log_f32_body` for `m - 1` and `m + 1` computations. |

### Address Mode Configuration

The erfinv operation uses `ADDR_MOD_7` configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::erfinv>()`. Since erfinv is not one of the special-cased types (topk_local_sort, typecast, unary_max/min, reciprocal), only the default ADDR_MOD_7 is set.

**ADDR_MOD_7 configuration (identical on Wormhole and Blackhole):**

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
```

All increment fields are zero. This means the SFPU does not auto-increment any source or destination register addresses between instructions -- all address advancement is handled explicitly by the SFPU kernel code via `sfpi::dst_reg++` (which compiles to `SFPINCRWC`) and by the params dispatch via `TTI_SETRWC` between faces.

The Wormhole and Blackhole implementations are identical for this operation. Both configure ADDR_MOD_7 with zero increments, and neither sets ADDR_MOD_6 (only done for topk, typecast, and unary_max/min operations, plus reciprocal on Blackhole).

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How is the erfinv SFPU kernel implemented? What is the call chain from the compute kernel API through LLK to the ckernel SFPU implementation for erfinv? What files contain the SFPU kernel for erfinv?"
   **Reason**: Needed to identify all file paths in the abstraction chain and understand the overall architecture of the erfinv SFPU dispatch.
   **Key Findings**: Identified the API header (`erfinv.h`), the macro expansion path (`SFPU_UNARY_NO_PARAM_KERNEL_FN`), the LLK params dispatch function, and the core SFPU implementation file. Confirmed the algorithm is based on Winitzki's 2008 approximation with dependencies on `calculate_log_body` and `calculate_sqrt_custom`.

2. **Query**: "How is the erfinv SFPU kernel implemented in the LLK layer? What is the ckernel_sfpu function for erfinv, and what SFPU instructions does it use?"
   **Reason**: Needed detail on the LLK-level dispatch, addr_mod configuration, and the `_llk_math_eltwise_unary_sfpu_init_`/`_llk_math_eltwise_unary_sfpu_params_` functions.
   **Key Findings**: Confirmed the addr_mod configuration uses ADDR_MOD_7 with all-zero increments. The params function handles face iteration with TTI_SETRWC instructions. The init function calls `_init_sfpu_config_reg()`, configures addr_mod, and resets counters.

### Confluence References

No Confluence references were consulted for this analysis. The SFPU ISA page was not needed because the erfinv kernel uses only standard SFPI intrinsics (arithmetic, conditional execution, register load/store) that are well-documented in the DeepWiki sources and the source code itself.

### Glean References

No Glean references were consulted for this analysis.
