---
name: ttnn-factory-patterns
description: Get guidance on TTNN program factory patterns (CB creation, kernel setup, work distribution, runtime args). Use this skill whenever working with code that creates Programs, configures circular buffers, creates kernels, or sets runtime args - including writing, reading, reviewing, improving, refactoring, or debugging. Always consult BEFORE making changes.
---

# TTNN Program Factory Patterns Expert

You are helping the user with TTNN program factory patterns - the host-side code that configures circular buffers, creates kernels, and distributes work across cores.

## Modern vs Legacy API

### Circular Buffer Creation

**MODERN API (preferred)**: `tt::tt_metal::create_cb()`
```cpp
#include "ttnn/operations/cb_utils.hpp"

// Single CB - returns tuple of (cb_index, handle)
auto [cb_id, cb_handle] = tt::tt_metal::create_cb(
    cb_index,           // uint32_t: CB index (e.g., tt::CBIndex::c_0)
    program,            // Program&
    core_spec,          // CoreRange, CoreRangeSet, or CoreCoord
    page_size,          // uint32_t: bytes per page
    num_pages,          // uint32_t: number of pages (buffering)
    data_format,        // tt::DataFormat
    buffer              // Buffer* (optional): for sharded tensors
);

// Multiple CBs sharing config - returns tuple of (array of cb_indices, handle)
uint32_t cbs[] = {tt::CBIndex::c_0, tt::CBIndex::c_1};
auto [cb_ids, cb_handle] = tt::tt_metal::create_cb(
    cbs, program, core_spec, page_size, num_pages, data_format
);
```

**LEGACY API (still works, but verbose)**:
```cpp
// Legacy - requires manual config setup
auto cb_config = CircularBufferConfig(num_pages * page_size, {{cb_id, data_format}})
    .set_page_size(cb_id, page_size);
auto cb_handle = CreateCircularBuffer(program, core_spec, cb_config);
```

### Why Modern API is Better
1. **Simpler** - single function call vs config object + creation
2. **Sharded support** - pass `buffer` directly for globally allocated CBs
3. **Type-safe** - returns CB index in tuple, no mismatch risk
4. **Multi-CB** - array overload for multiple CBs with same config

## Const Correctness (CRITICAL)

**Every variable that can be `const` SHOULD be `const`.** This includes:
- Tensor references
- Buffer pointers
- Data formats
- Tile sizes
- Loop bounds
- Anything computed once and never modified

**Use `auto*` for pointers** - Makes pointer types explicit and catches errors:
```cpp
// WRONG - hides pointer type
auto src_buffer = input.buffer();

// CORRECT - explicit pointer
const auto* src_buffer = input.buffer();
```

```cpp
// WRONG - mutable when it shouldn't be
auto input = inputs.input_tensor;
auto in_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
uint32_t tile_size = tt::tile_size(in_df);
uint32_t Ht = input.padded_shape()[2] / tt::constants::TILE_HEIGHT;

// CORRECT - const everything, auto* for pointers
const auto& input = inputs.input_tensor;
const auto* src_buffer = input.buffer();
const tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
const uint32_t tile_size = tt::tile_size(in_df);
const uint32_t Ht = input.padded_shape()[2] / tt::constants::TILE_HEIGHT;
```

## Program Factory Structure

### Required Includes
```cpp
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
```

### Standard Factory Template
```cpp
namespace ttnn::operations::my_op {

MyOp::ProgramFactory::cached_program_t MyOp::ProgramFactory::create(
    const operation_attributes_t& attrs,
    const tensor_args_t& inputs,
    tensor_return_value_t& output)
{
    // ============================================================
    // CONSTANTS - Define buffering factor and other constants FIRST
    // ============================================================
    constexpr uint32_t buffering_factor = 2;  // Double buffering (MUST be constexpr)

    // ============================================================
    // 1. Extract tensor properties (ALL const)
    // ============================================================
    const auto& input = inputs.input_tensor;
    const tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t tile_size_in = tt::tile_size(in_df);
    const uint32_t tile_size_out = tt::tile_size(out_df);

    // Tensor dimensions (const)
    const uint32_t num_tiles = input.volume() / tt::constants::TILE_HW;
    const uint32_t Ht = input.padded_shape()[2] / tt::constants::TILE_HEIGHT;
    const uint32_t Wt = input.padded_shape()[3] / tt::constants::TILE_WIDTH;

    // ============================================================
    // 2. Create program
    // ============================================================
    Program program = Program();

    // ============================================================
    // 3. Work distribution (const results)
    // ============================================================
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2,
                tiles_per_core_g1, tiles_per_core_g2] =
        tt::tt_metal::split_work_to_cores(grid, num_tiles);

    // ============================================================
    // 4. Create circular buffers
    //    - Apply buffering_factor to intermediate/local CBs
    //    - Do NOT apply to globally allocated (sharded) CBs
    // ============================================================

    // Input CB - apply buffering_factor (not globally allocated)
    tt::tt_metal::create_cb(tt::CBIndex::c_0, program, all_cores,
                            tile_size_in, buffering_factor, in_df);

    // Intermediate CB - apply buffering_factor
    tt::tt_metal::create_cb(tt::CBIndex::c_24, program, all_cores,
                            tile_size_in, buffering_factor, in_df);

    // Output CB - apply buffering_factor (not globally allocated)
    tt::tt_metal::create_cb(tt::CBIndex::c_16, program, all_cores,
                            tile_size_out, buffering_factor, out_df);

    // Scaler CB - single page, no buffering needed
    tt::tt_metal::create_cb(tt::CBIndex::c_1, program, all_cores,
                            tile_size_in, 1, in_df);

    // ============================================================
    // 5. Create kernels
    // ============================================================
    const std::vector<uint32_t> reader_compile_args = {/* compile args */};
    const auto reader_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/my_op/device/kernels/reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    const std::vector<uint32_t> compute_compile_args = {tiles_per_core_g1};
    const auto compute_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/my_op/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .compile_args = compute_compile_args});

    const std::vector<uint32_t> writer_compile_args = {/* compile args */};
    const auto writer_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/my_op/device/kernels/writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // ============================================================
    // 6. Set runtime args per core
    // ============================================================
    const auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t tile_offset = 0;  // This one mutates in loop
    for (const auto& core : cores) {
        const uint32_t num_tiles_this_core = core_group_1.contains(core)
            ? tiles_per_core_g1 : tiles_per_core_g2;

        SetRuntimeArgs(program, reader_id, core,
            {input.buffer()->address(), num_tiles_this_core, tile_offset});
        SetRuntimeArgs(program, writer_id, core,
            {output.buffer()->address(), num_tiles_this_core, tile_offset});

        tile_offset += num_tiles_this_core;
    }

    // ============================================================
    // 7. Return cached program
    // ============================================================
    return {std::move(program), {reader_id, compute_id, writer_id, cores}};
}

} // namespace
```

## Work Distribution

### split_work_to_cores()
```cpp
#include <tt-metalium/work_split.hpp>

auto [num_cores,        // Total cores used
      all_cores,        // CoreRangeSet of all cores
      core_group_1,     // CoreRangeSet with more work
      core_group_2,     // CoreRangeSet with less work (may be empty)
      work_per_core_g1, // Work units for group 1
      work_per_core_g2  // Work units for group 2
     ] = tt::tt_metal::split_work_to_cores(grid, total_work);
```

### Manual Core Iteration
```cpp
auto cores = corerange_to_cores(all_cores, std::nullopt);
for (const auto& core : cores) {
    uint32_t work_this_core;
    if (core_group_1.contains(core)) {
        work_this_core = work_per_core_g1;
    } else if (core_group_2.contains(core)) {
        work_this_core = work_per_core_g2;
    } else {
        continue;  // Or set to 0 for no-op
    }
    // Set runtime args...
}
```

## Kernel Creation

Use `tt::tt_metal::CreateKernel` with the appropriate config:

```cpp
// Reader kernel
const std::vector<uint32_t> reader_compile_args = {arg1, arg2};
const auto reader_id = tt::tt_metal::CreateKernel(
    program,
    "path/to/reader.cpp",
    all_cores,
    tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

// Compute kernel
const std::vector<uint32_t> compute_compile_args = {tiles_per_core};
const auto compute_id = tt::tt_metal::CreateKernel(
    program,
    "path/to/compute.cpp",
    all_cores,
    tt::tt_metal::ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .compile_args = compute_compile_args});

// Writer kernel
const std::vector<uint32_t> writer_compile_args = {arg1, arg2};
const auto writer_id = tt::tt_metal::CreateKernel(
    program,
    "path/to/writer.cpp",
    all_cores,
    tt::tt_metal::WriterDataMovementConfig(writer_compile_args));
```

## CB Configuration Patterns

### Explicit CB Parameter Naming (Code Style)

**Every CB should have clearly named variables for its configuration.** Even if slightly more verbose, explicit names make the code self-documenting and easier to understand at a glance.

```cpp
// WRONG - magic numbers, hard to understand intent
tt::tt_metal::create_cb(tt::CBIndex::c_0, program, all_cores, stick_size, 64, cb_data_format);
tt::tt_metal::create_cb(tt::CBIndex::c_1, program, all_cores, tile_size, 2 * Wt, cb_data_format);

// CORRECT - explicit names explain the configuration
// CB c_0: Input RM sticks (double-buffered, 32 sticks per tile row)
constexpr uint32_t cb_in_rm_idx = tt::CBIndex::c_0;
const uint32_t cb_in_rm_page_size = stick_size;
const uint32_t cb_in_rm_num_pages = buffering_factor * tt::constants::TILE_HEIGHT;  // 2 * 32 = 64
tt::tt_metal::create_cb(cb_in_rm_idx, program, all_cores, cb_in_rm_page_size, cb_in_rm_num_pages, cb_data_format);

// CB c_1: Tiled input (double-buffered)
constexpr uint32_t cb_in_tiled_idx = tt::CBIndex::c_1;
const uint32_t cb_in_tiled_page_size = tile_size;
const uint32_t cb_in_tiled_num_pages = buffering_factor * Wt;
tt::tt_metal::create_cb(cb_in_tiled_idx, program, all_cores, cb_in_tiled_page_size, cb_in_tiled_num_pages, cb_data_format);
```

**Naming convention:**
- `cb_<name>_idx` - CB index (constexpr)
- `cb_<name>_page_size` - Size of one page in bytes
- `cb_<name>_num_pages` - Number of pages (buffering depth)
- `input_stick_size` / `output_stick_size` - Never just "stick_size" (be explicit about which tensor)

This pattern makes it immediately clear:
1. Which CB index is being configured
2. What the page size represents (stick, tile, etc.)
3. Why the number of pages was chosen (buffering, holding full row, etc.)

### Stick Size and NoC Alignment

**NEVER hardcode alignment values.** Use the buffer's `alignment()` method:

```cpp
// Naming convention:
// - *_stick_size = raw/unaligned size
// - *_stick_size_aligned = aligned for NoC transfers

// WRONG - hardcoded alignment
constexpr uint32_t noc_alignment = 32;  // NEVER DO THIS
const uint32_t input_stick_size_aligned =
    ((input_stick_size + noc_alignment - 1) / noc_alignment) * noc_alignment;

// CORRECT - use buffer's alignment() method
const uint32_t input_stick_size = W * input.element_size();
const uint32_t input_stick_size_aligned = tt::round_up(input_stick_size, src_buffer->alignment());

// Use aligned version for CB page sizes and kernel args (NoC transfers)
const uint32_t cb_in_rm_page_size = input_stick_size_aligned;
std::vector<uint32_t> reader_compile_args = {input_stick_size_aligned, Wt};
```

**Buffer alignment methods:**
- `buffer->alignment()` - get alignment requirement for this buffer
- `buffer->aligned_page_size()` - get pre-aligned page size
- `tt::round_up(size, alignment)` - align a size to given alignment

**Always use explicit names for stick sizes:**
```cpp
// WRONG - ambiguous
const uint32_t stick_size = W * element_size;

// CORRECT - explicit about which tensor
const uint32_t input_stick_size = W * input.element_size();
const uint32_t input_stick_size_aligned = tt::round_up(input_stick_size, src_buffer->alignment());
const uint32_t output_stick_size = W * output.element_size();
const uint32_t output_stick_size_aligned = tt::round_up(output_stick_size, dst_buffer->alignment());
```

### Buffering Factor Rules

**CRITICAL**: Define `buffering_factor` as `constexpr` at the top of the factory:
```cpp
constexpr uint32_t buffering_factor = 2;  // Double buffering
```

| CB Type | Apply buffering_factor? | Reason |
|---------|------------------------|--------|
| Interleaved input | YES | Enables overlap |
| Interleaved output | YES | Enables overlap |
| Intermediate | YES | Enables pipelining |
| Scaler | NO (use 1) | Read once, used many times |
| **Globally allocated (sharded)** | **NO** | Size determined by shard spec |

### Standard Interleaved Input/Output
```cpp
constexpr uint32_t buffering_factor = 2;

// Input CB - apply buffering_factor
tt::tt_metal::create_cb(tt::CBIndex::c_0, program, all_cores,
    tile_size, buffering_factor, in_df);

// Output CB - apply buffering_factor
tt::tt_metal::create_cb(tt::CBIndex::c_16, program, all_cores,
    tile_size, buffering_factor, out_df);
```

### Sharded Input/Output (Globally Allocated)
```cpp
// IMPORTANT: Do NOT apply buffering_factor to globally allocated CBs
// The num_pages is determined by the shard spec, not buffering strategy

const uint32_t shard_num_pages = input.shard_spec().value().shape[0];
const uint32_t shard_page_size = input.shard_spec().value().shape[1] * input.element_size();

// Sharded input - num_pages from shard spec, NOT buffering_factor
auto [cb_in, cb_handle] = tt::tt_metal::create_cb(
    tt::CBIndex::c_0, program, all_cores,
    shard_page_size, shard_num_pages, in_df,
    input.buffer());  // Pass buffer for globally allocated

// Sharded output - same pattern
const uint32_t out_shard_pages = output.shard_spec().value().shape[0];
auto [cb_out, cb_out_handle] = tt::tt_metal::create_cb(
    tt::CBIndex::c_16, program, all_cores,
    out_page_size, out_shard_pages, out_df,
    output.buffer());
```

### Intermediate CB (compute internal)
```cpp
// Intermediate for multi-phase compute - apply buffering_factor
tt::tt_metal::create_cb(tt::CBIndex::c_24, program, all_cores,
    tile_size, buffering_factor, df);

// Or if you need to hold multiple tiles for reduction:
const uint32_t tiles_to_hold = Wt;  // e.g., hold full row for row reduction
tt::tt_metal::create_cb(tt::CBIndex::c_24, program, all_cores,
    tile_size, tiles_to_hold, df);
```

### Scaler CB (small, read-once)
```cpp
// Scaler tile - always 1 page, no buffering needed
tt::tt_metal::create_cb(tt::CBIndex::c_1, program, all_cores,
    tile_size, 1, df);
```

## Common CB Index Conventions

| Index | Constant | Typical Use |
|-------|----------|-------------|
| 0 | `tt::CBIndex::c_0` | Primary input |
| 1 | `tt::CBIndex::c_1` | Secondary input / scaler |
| 2 | `tt::CBIndex::c_2` | Tertiary input |
| 16 | `tt::CBIndex::c_16` | Primary output |
| 24 | `tt::CBIndex::c_24` | First intermediate |
| 25 | `tt::CBIndex::c_25` | Second intermediate |

## Runtime Args Patterns

### Basic Per-Core Args
```cpp
SetRuntimeArgs(program, kernel_id, core, {
    buffer_address,
    num_tiles,
    tile_offset
});
```

### Using TensorAccessorArgs
```cpp
#include <tt-metalium/tensor_accessor_args.hpp>

std::vector<uint32_t> compile_args = {cb_index, elems_per_page, page_size};
tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(compile_args);
```

## Compute Kernel Config

### Extracting from DeviceComputeKernelConfig
```cpp
auto [math_fidelity, math_approx_mode, fp32_dest_acc_en,
      packer_l1_acc, dst_full_sync_en] =
    get_compute_kernel_config_args(device->arch(), compute_kernel_config);
```

### Setting Defines for Compute
```cpp
std::map<std::string, std::string> compute_defines;
if (fp32_dest_acc_en) {
    compute_defines["FP32_DEST_ACC_EN"] = "1";
}
compute_defines["REDUCE_OP"] = "PoolType::SUM";  // deprecated, use templates
```

## Common Mistakes

### 1. Missing const Qualifiers
```cpp
// WRONG - unnecessarily mutable
auto& input = inputs.input_tensor;
auto in_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
uint32_t tile_size = tt::tile_size(in_df);

// CORRECT - const everything that doesn't change
const auto& input = inputs.input_tensor;
const tt::DataFormat in_df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
const uint32_t tile_size = tt::tile_size(in_df);
```

### 2. Hardcoded Buffering Factor
```cpp
// WRONG - magic number scattered throughout
tt::tt_metal::create_cb(cb_in, program, cores, tile_size, 2, df);
tt::tt_metal::create_cb(cb_out, program, cores, tile_size, 2, df);
tt::tt_metal::create_cb(cb_intermed, program, cores, tile_size, 2, df);

// CORRECT - single constexpr definition
constexpr uint32_t buffering_factor = 2;
tt::tt_metal::create_cb(cb_in, program, cores, tile_size, buffering_factor, df);
tt::tt_metal::create_cb(cb_out, program, cores, tile_size, buffering_factor, df);
tt::tt_metal::create_cb(cb_intermed, program, cores, tile_size, buffering_factor, df);
```

### 3. Applying Buffering Factor to Globally Allocated CBs
```cpp
// WRONG - buffering_factor on sharded CB
constexpr uint32_t buffering_factor = 2;
tt::tt_metal::create_cb(cb_in, program, cores, page_size, buffering_factor, df,
    input.buffer());  // BUG: globally allocated, size comes from shard spec!

// CORRECT - use shard spec for globally allocated
const uint32_t shard_pages = input.shard_spec().value().shape[0];
tt::tt_metal::create_cb(cb_in, program, cores, page_size, shard_pages, df,
    input.buffer());
```

### 4. Forgetting to Set Args for All Cores
```cpp
// WRONG - crashes if core not in group
for (const auto& core : cores) {
    uint32_t work = core_group_1.contains(core) ? work_g1 : work_g2;
    SetRuntimeArgs(program, kernel_id, core, {work});
}

// CORRECT - handle all cases
for (const auto& core : cores) {
    uint32_t work;
    if (core_group_1.contains(core)) {
        work = work_g1;
    } else if (core_group_2.contains(core)) {
        work = work_g2;
    } else {
        work = 0;  // No-op for unused cores
    }
    SetRuntimeArgs(program, kernel_id, core, {work});
}
```

### 5. Mismatched Page Sizes
```cpp
// WRONG - CB page size != kernel expectation
tt::tt_metal::create_cb(cb_id, program, cores, TILE_SIZE, buffering_factor, df);
// But kernel uses get_tile_size() which might differ

// CORRECT - use consistent tile_size calculation
const uint32_t tile_size = tt::tile_size(df);
tt::tt_metal::create_cb(cb_id, program, cores, tile_size, buffering_factor, df);
```

### 6. Wrong Data Format Conversion
```cpp
// WRONG - DataType is not DataFormat
auto df = input.dtype();  // This is DataType, not DataFormat!

// CORRECT
const tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
```

### 7. Hardcoded Alignment Values
```cpp
// WRONG - hardcoded alignment
constexpr uint32_t noc_alignment = 32;  // NEVER hardcode alignment!
const uint32_t aligned_size = ((size + noc_alignment - 1) / noc_alignment) * noc_alignment;

// CORRECT - use buffer's alignment method
const uint32_t aligned_size = tt::round_up(size, buffer->alignment());
```

## Workflow

1. **Define constants first** - `constexpr uint32_t buffering_factor = 2;`
2. **Make everything const** - tensor refs, data formats, dimensions, kernel handles
3. **Read existing factories** for similar operations as reference
4. **Use modern `create_cb` API** from `cb_utils.hpp`
5. **Apply buffering_factor consistently** - except for globally allocated CBs
6. **Use `split_work_to_cores`** for work distribution
7. **Use `tt::tt_metal::CreateKernel`** with appropriate config (Reader/Writer/Compute)
8. **Set runtime args for ALL cores** in the core range
9. **Match CB page sizes** between factory and kernel expectations
