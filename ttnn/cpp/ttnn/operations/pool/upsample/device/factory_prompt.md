# Program Factory Implementation Template

## Objective
Implement a new TTNN operation program factory for operations on multi-dimensional tensors in **row-major DRAM interleaved layout**.

### This Template's Example: Upsample 3D
This document uses **3D upsample** with nearest-neighbor interpolation for 5D tensors (N, D, H, W, C) as a concrete example, but the methodology applies to other operations.

## General Requirements

### Start with the Program Factory
Begin by designing and implementing the **program factory** (C++ implementation), as this is the critical component where:
- **Data flow** is orchestrated between DRAM, circular buffers, and compute units
- **Tensor layout decisions** are made (how work is distributed and indexed)
- **Circular buffers** are configured for inter-kernel communication
- **Kernels** are instantiated and coordinated (reader, writer, compute)
- **Memory access patterns** are defined

**Do NOT proceed with kernel implementation or Python bindings until the factory design is agreed upon and validated.**

## Understanding Tensor Layout

### Channel-Last Layout and Work Distribution
For operations on tensors with channel-last layout (e.g., N×D×H×W×C):
- **Flattened view**: Treat the tensor as a 2D array with spatial dimensions flattened into rows
  - Example: 5D tensor (N, D, H, W, C) → N×D×H×W rows, each with C elements
- **Work unit**: Typically one "stick" (one row of C contiguous elements in memory)
- **Total work units**: Product of all spatial dimensions (N × D × H × W in the example)
- **Work distribution**: Work units are split among available cores, with each core processing a contiguous chunk

This pattern is common in TTNN operations because:
- Channels are the innermost (fastest-changing) dimension in memory
- Processing one row at a time provides good cache locality
- Work is naturally parallelizable across spatial dimensions

## Three-Stage Implementation Process

### Stage 1: Analyze Reference Implementation
Study an existing similar operation to understand TTNN patterns. Document findings in `<operation>_context.MD`:

**Key concepts to extract**:
- **Work unit definition**: What constitutes one unit of work? (row, tile, block?)
- **Circular buffer management**: How are CBs sized and allocated?
- **Data flow pattern**: How does data move through reader → CB → compute → writer?
- **Index calculations**: How are tensor indices mapped to/from linear memory indices?
- **Memory access patterns**: What order is data read/written?
- **Layout-specific handling**: Differences between row-major, tiled, sharded layouts
- **Core distribution**: How is work split across available cores?
- **Compile-time vs runtime arguments**: What parameters are fixed vs dynamic?

### Example Reference for This Template
For the upsample3d example, study:
- **File**: `ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_interleaved.cpp`
- Focus on understanding the 2D nearest-neighbor pattern to extend it to 3D

### Stage 2: Create Implementation Plan
Produce a detailed plan in `<operation>_factoryPlan.MD`:

**Plan should address**:
1. **Work distribution strategy**:
   - How to define work units for your operation
   - How many work units total?
   - How to handle each tensor dimension?

2. **Circular buffer requirements**:
   - What buffers are needed? (input, output, intermediate)
   - Size per buffer based on work unit
   - Double-buffering considerations
   - Alignment requirements

3. **Index calculation strategy**:
   - Linear index → N-dimensional coordinates
   - N-dimensional coordinates → output positions
   - Handle any special indexing patterns for your operation

4. **Kernel architecture**:
   - Can existing kernels be reused?
   - What new kernels are needed?
   - What does each kernel do?

5. **Argument structure**:
   - Compile-time arguments (fixed at kernel creation)
   - Runtime arguments (change per invocation)
   - What information does each kernel need?

6. **Key differences from reference**:
   - What's different about your operation?
   - Additional dimensions, computations, or complexity?

### Stage 3: Implement Program Factory
Create the following files:
- `<operation>_program_factory.hpp`: Function declaration and documentation
- `<operation>_program_factory.cpp`: Full implementation

## Program Factory Structure

A typical program factory implementation includes:

### 1. Validation and Setup
```cpp
// Validate input tensor properties
TT_FATAL(input.get_shape().rank() == expected_rank, "Error message");
TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Layout error");
TT_FATAL(input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Memory error");

// Extract dimensions and calculate work parameters
const auto& input_shape = input.padded_shape();
// ... extract relevant dimensions
```

### 2. Work Distribution Calculation
```cpp
// Define work unit size and count
const uint32_t input_unit_size = <innermost_dim> * element_size;
const uint32_t aligned_input_unit_size = round_up(input_unit_size, dram_alignment);
const uint32_t work_units_to_split = <total_work_units>;

// Distribute across cores
const auto [num_cores, all_cores, core_group_1, core_group_2,
            work_per_core_group_1, work_per_core_group_2] =
    split_work_to_cores(compute_grid_size, work_units_to_split);
```

### 3. Circular Buffer Creation
```cpp
// Determine CB size (consider double-buffering)
uint32_t num_pages_in_cb = <pages_per_work_unit>;
if (work_per_core_group_1 > 1) {
    num_pages_in_cb *= 2;  // Double buffer for pipelining
}

// Create buffers
const auto [cb_index, cb_handle] = create_cb(
    next_cb_index++, program, all_cores,
    aligned_unit_size, num_pages_in_cb, data_format);
```

### 4. Kernel Instantiation
```cpp
// Reader kernel - loads data from DRAM to CB
std::vector<uint32_t> reader_compile_args = {cb_index, unit_size, ...};
TensorAccessorArgs(src_buffer).append_to(reader_compile_args);
auto reader_kernel_id = CreateKernel(program, "path/to/reader.cpp",
                                     all_cores, ReaderDataMovementConfig(reader_compile_args));

// Writer kernel - writes data from CB to DRAM
std::vector<uint32_t> writer_compile_args = {cb_index, unit_size, <operation_params>, ...};
TensorAccessorArgs(dst_buffer).append_to(writer_compile_args);
auto writer_kernel_id = CreateKernel(program, "path/to/writer.cpp",
                                     all_cores, WriterDataMovementConfig(writer_compile_args));

// Compute kernel (if needed) - processes data in CB
// ...
```

### 5. Runtime Arguments Configuration
```cpp
// Set arguments for each core
for (uint32_t i = 0, units_processed = 0; i < num_cores; i++) {
    const CoreCoord core = {i / num_cores_y, i % num_cores_y};
    uint32_t units_per_core = /* determine from core groups */;

    std::vector<uint32_t> reader_rt_args = {
        src_buffer->address(),
        units_per_core,      // how many units
        units_processed      // starting unit ID
    };

    std::vector<uint32_t> writer_rt_args = {
        dst_buffer->address(),
        units_per_core,
        units_processed
    };

    SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);
    SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

    units_processed += units_per_core;
}
```

### 6. Override Callback
```cpp
// Allow buffer address updates when tensors move
auto override_callback = [kernel_ids...](const void* operation,
                                         Program& program,
                                         const std::vector<Tensor>& input_tensors,
                                         const std::vector<std::optional<const Tensor>>&,
                                         const std::vector<Tensor>& output_tensors) {
    // Update buffer addresses in runtime args for all cores
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        auto& reader_args = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_args[0] = input_tensors[0].buffer()->address();
        // ... update other kernels
    }
};

return {.program = std::move(program),
        .override_runtime_arguments_callback = override_callback};
```

## Key Design Considerations

### Work Unit Granularity
- **Too small**: High overhead from kernel dispatch and synchronization
- **Too large**: Poor load balancing, underutilized cores
- **Sweet spot**: Typically one row/stick for row-major data (good locality, easy parallelization)

### Circular Buffer Sizing
- Must fit at least one work unit
- Double-buffering improves throughput (overlap compute and data movement)
- Consider L1 memory constraints (1.5MB per Tensix core)

### Memory Alignment
- DRAM accesses must be aligned (typically 16 or 32 bytes)
- Use `round_up(size, hal::get_dram_alignment())` for CB sizes
- TensorAccessor handles page-aligned access patterns

### Index Calculations
- Validate index math thoroughly - this is error-prone
- Consider stride patterns for efficiency
- Test with small tensors first to verify correctness

### Kernel Reuse
- Check if existing kernels can be reused (especially readers for simple patterns)
- Reader kernels often reusable for sequential stick reading
- Writer/compute kernels typically need operation-specific logic

## Expected Deliverables

1. **`<operation>_context.MD`**: Analysis of reference implementation
2. **`<operation>_factoryPlan.MD`**: Detailed implementation plan
3. **`<operation>_program_factory.hpp`**: Header with function declarations
4. **`<operation>_program_factory.cpp`**: Full factory implementation

## Success Criteria

The program factory implementation should:
- ✅ Correctly validate input tensor properties
- ✅ Calculate and distribute work units appropriately across cores
- ✅ Create circular buffers with proper sizing and alignment
- ✅ Instantiate kernels with correct compile-time arguments
- ✅ Configure runtime arguments correctly for each core
- ✅ Provide override callback for buffer address updates
- ✅ Handle index calculations correctly
- ✅ Follow TTNN coding conventions and patterns
- ✅ Be ready for integration with kernel implementations

## Common Pitfalls to Avoid

1. **Incorrect work distribution**: Forgetting to account for uneven work splits (core_group_1 vs core_group_2)
2. **Alignment issues**: Not rounding CB sizes to DRAM alignment
3. **Index math errors**: Off-by-one errors in N-D to linear conversions
4. **Missing validation**: Not checking tensor rank, layout, memory config
5. **Argument structure**: Mixing up compile-time vs runtime arguments
6. **Buffer reuse**: Incorrectly sharing CBs between incompatible operations
7. **Memory overflow**: CB sizes exceeding L1 capacity

## Example: Upsample 3D Specifics

For the upsample3d operation as a concrete example:

### Tensor Specifications
- **Input**: 5D tensor (N, D, H, W, C)
- **Output**: 5D tensor (N, D×scale_d, H×scale_h, W×scale_w, C)
- **Operation**: Nearest-neighbor interpolation (each input element replicated scale_d × scale_h × scale_w times)

### Work Distribution
- Work units: N × D × H × W sticks (rows of C elements)
- Each input stick written to scale_d × scale_h × scale_w output locations

### Index Calculation Example
```cpp
// Linear stick index → 5D input coordinates
uint32_t sticks_per_batch = D * H * W;
uint32_t n = stick_index / sticks_per_batch;
uint32_t remainder = stick_index % sticks_per_batch;
uint32_t w = remainder % W;
uint32_t h = (remainder / W) % H;
uint32_t d = remainder / (W * H);

// For each output position in the upsampled cube
for (uint32_t sd = 0; sd < scale_d; sd++)
  for (uint32_t sh = 0; sh < scale_h; sh++)
    for (uint32_t sw = 0; sw < scale_w; sw++)
      output_stick_index = n * (D*scale_d*H*scale_h*W*scale_w) +
                          (d*scale_d + sd) * (H*scale_h*W*scale_w) +
                          (h*scale_h + sh) * (W*scale_w) +
                          (w*scale_w + sw);
```

### Special Considerations for Upsample
- **Write amplification**: Each input written multiple times (scale_d × scale_h × scale_w)
- **Reader reuse**: Can reuse existing 2D reader since it just reads sticks sequentially
- **Writer complexity**: Needs 3D upsampling logic with nested loops
