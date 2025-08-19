# Upsample 3D Height Sharded Implementation Plan

## Overview
This document outlines the plan to extend the existing Upsample 3D operation to support height sharded input and output tensors using TensorAccessor. The implementation will follow strict Test-Driven Development (TDD) principles.

## Current State Analysis

### Existing Implementation
- **Location**: `ttnn/cpp/ttnn/operations/pool/upsample3d/`
- **Current Support**: Row major interleaved tensors only
- **Key Files**:
  - `upsample3d.hpp`: Main operation interface
  - `device/upsample3d_op.hpp`: Device operation structure
  - `device/upsample3d_program_factory.cpp`: Program factory with TensorAccessor
  - Kernels in `device/kernels/dataflow/` and `device/kernels/compute/`

### Reference Implementation
- **Location**: `ttnn/cpp/ttnn/operations/pool/upsample/device/upsample_program_factory_multicore_sharded.cpp`
- **Features**: Height sharded and block sharded support
- **Key Components**:
  - Config tensor creation for shard mapping
  - Complex work distribution logic
  - Reader/writer kernels for sharded data

## Implementation Strategy

### Phase 1: Test Infrastructure Setup
**Objective**: Create comprehensive test suite for height sharded tensors

#### Tests to Add:
1. **Basic Height Sharded Tests**
   ```python
   # Use ttnn.create_sharded_memory_config for proper height sharding
   height_sharded_config = ttnn.create_sharded_memory_config(
       shape=[1, 2, 8, 8, 32],  # NDHWC format
       core_grid=ttnn.CoreGrid(y=4, x=8),  # Height sharding across 32 cores (4x8 grid)
       strategy=ttnn.ShardStrategy.HEIGHT,
       orientation=ttnn.ShardOrientation.ROW_MAJOR,
       use_height_and_width_as_shard_shape=False,
   )
   ```
   - Single core height sharded input/output
   - Multi-core height sharded with simple shard shapes
   - Various scale factors (1,1,1), (2,2,2), (1,2,1)

2. **Shard Shape Compatibility Tests**
   ```python
   # Use device's actual compute grid for realistic testing
   device_grid = device.compute_with_storage_grid_size()

   # Different core grid configurations - height sharding uses all cores
   configs = [
       ttnn.CoreGrid(y=1, x=2),                    # 2 cores height sharded
       ttnn.CoreGrid(y=2, x=2),                    # 4 cores height sharded
       ttnn.CoreGrid(y=min(2, device_grid.y), x=min(4, device_grid.x)),  # 8 cores (device dependent)
       ttnn.CoreGrid(y=device_grid.y, x=device_grid.x),  # Full device grid
   ]
   ```
   - Input shard shapes that divide evenly
   - Edge cases with remainder shards
   - Different number of cores (2, 4, 8, 16)

3. **Scale Factor Variation Tests**
   - Asymmetric scaling with height sharding
   - Large scale factors (3x, 4x)
   - Mixed scaling patterns

4. **Memory Layout Tests**
   - Height sharded input → Height sharded output
   - Height sharded input → Interleaved output (fallback)
   - Validation of shard specs matching

#### Test Structure:
```python
@pytest.mark.parametrize("num_cores_factor", [2, 4, 8, "full"])  # Different core counts
@pytest.mark.parametrize("input_shape_multiplier", [1, 2, 4])     # Different tensor sizes
@pytest.mark.parametrize("scale_factor", [(2,2,2), (1,2,1), (3,1,2)])
@pytest.mark.timeout(30)  # Tight timeout to catch hanging tests
def test_upsample3d_height_sharded(device, num_cores_factor, input_shape_multiplier, scale_factor):
    # Get device's actual compute grid
    device_grid = device.compute_with_storage_grid_size()

    # Calculate core grid based on device capabilities
    if num_cores_factor == "full":
        core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)
    elif num_cores_factor == 2:
        core_grid = ttnn.CoreGrid(y=1, x=2)
    elif num_cores_factor == 4:
        core_grid = ttnn.CoreGrid(y=2, x=2)
    elif num_cores_factor == 8:
        core_grid = ttnn.CoreGrid(y=min(2, device_grid.y), x=min(4, device_grid.x))

    # Create input shape based on multiplier and ensure divisibility by core count
    base_dhw = 8 * input_shape_multiplier  # Ensure good divisibility
    input_shape_ndhwc = [1, base_dhw, base_dhw, base_dhw, 32]

    # Create height sharded memory config using device grid
    height_sharded_config = ttnn.create_sharded_memory_config(
        shape=input_shape_ndhwc,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    # Test implementation...
```

### Phase 2: Core Implementation
**Objective**: Implement height sharded support using TensorAccessor

#### Changes Required:

1. **Operation Validation (`upsample3d_op.cpp`)**
   - Add height sharded memory layout validation
   - Verify input/output shard specs compatibility
   - Check that scale factors work with shard dimensions

2. **Program Factory Extension (`upsample3d_program_factory.cpp`)**
   - Add new function: `upsample3d_multi_core_height_sharded()`
   - Remove dependency on config tensor (as specified)
   - Implement work distribution from output tensor perspective
   - Use TensorAccessor for both input and output tensors
   - **Single kernel approach** - no separate reader/writer kernels needed

3. **Single Kernel Implementation**
   - **One kernel only**: `writer_upsample3d_height_sharded.cpp` (similar to 2D)
   - Uses `is_reader` flag to split work between NCRISC and BRISC
   - Direct shard-to-shard data transfer using TensorAccessor
   - No config tensor - pure TensorAccessor approach

#### Key Implementation Details:

**Work Distribution Strategy**:
```cpp
// Work distribution based on output tensor shards
const auto output_shard_spec = output.shard_spec().value();
const auto input_shard_spec = input.shard_spec().value();

// Calculate work per core based on output shard dimensions
uint32_t output_shards_per_core = output_shard_spec.shape[0] / scale_factor_h;
uint32_t work_units_per_core = output_shards_per_core * scale_factor_d * scale_factor_w;
```

**Single Kernel Setup with TensorAccessor**:
```cpp
// Single kernel with both input and output TensorAccessor (no config tensor!)
std::vector<uint32_t> kernel_compile_time_args = {
    (std::uint32_t)in_cb_index,
    (std::uint32_t)out_cb_index,
    (std::uint32_t)is_reader,    // NCRISC=true, BRISC=false
    (std::uint32_t)stick_nbytes,
    (std::uint32_t)scale_factor_d,
    (std::uint32_t)scale_factor_h,
    (std::uint32_t)scale_factor_w
};

// Add input TensorAccessor args
tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(kernel_compile_time_args);
// Add output TensorAccessor args
tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(kernel_compile_time_args);

// Create single kernel for both NCRISC and BRISC
auto kernel_id = CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/pool/upsample3d/device/kernels/dataflow/writer_upsample3d_height_sharded.cpp",
    all_cores,
    DataMovementConfig(kernel_compile_time_args)
);
```

### Phase 3: Single Kernel Implementation
**Objective**: Implement single height sharded data movement kernel (similar to upsample 2D)

#### Key Insight from Upsample 2D:
The existing `writer_upsample_multi_core_sharded.cpp` uses a **single kernel** that:
- Acts as both reader and writer (no separate kernels needed)
- Uses `is_reader` compile-time flag to split work between NCRISC and BRISC
- Reads directly from input shards and writes to output shards
- Uses TensorAccessor instead of config tensor (our goal!)

#### Single Kernel Implementation (`writer_upsample3d_height_sharded.cpp`):
```cpp
void kernel_main() {
    // Compile-time arguments (no config tensor!)
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t is_reader = get_compile_time_arg_val(2); // NCRISC vs BRISC

    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t scale_factor_d = get_compile_time_arg_val(4);
    constexpr uint32_t scale_factor_h = get_compile_time_arg_val(5);
    constexpr uint32_t scale_factor_w = get_compile_time_arg_val(6);
    constexpr uint32_t output_d = get_compile_time_arg_val(7);
    constexpr uint32_t output_h = get_compile_time_arg_val(8);
    constexpr uint32_t output_w = get_compile_time_arg_val(9);

    // TensorAccessor setup for input and output (no config needed!)
    constexpr auto input_args = TensorAccessorArgs<10>();
    constexpr auto output_args = TensorAccessorArgs<14>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    uint32_t output_addr = get_arg_val<uint32_t>(1);
    uint32_t num_output_shards = get_arg_val<uint32_t>(2);
    uint32_t start_output_shard_id = get_arg_val<uint32_t>(3);

    const auto input_accessor = TensorAccessor(input_args, input_addr, stick_nbytes);
    const auto output_accessor = TensorAccessor(output_args, output_addr, stick_nbytes);

    // Calculate input dimensions
    const uint32_t input_d = output_d / scale_factor_d;
    const uint32_t input_h = output_h / scale_factor_h;
    const uint32_t input_w = output_w / scale_factor_w;

    // Split work between NCRISC and BRISC based on OUTPUT shards
    uint32_t output_shards_per_reader = (num_output_shards + 1) / 2;
    uint32_t reader_start_shard = start_output_shard_id;
    if (!is_reader) {
        reader_start_shard += output_shards_per_reader;
        output_shards_per_reader = num_output_shards - output_shards_per_reader;
    }

    // WORK FROM OUTPUT PERSPECTIVE: Process each output shard
    for (uint32_t shard_idx = 0; shard_idx < output_shards_per_reader; ++shard_idx) {
        uint32_t output_shard_id = reader_start_shard + shard_idx;

        // Iterate through pages in OUTPUT shard
        auto output_shard_pages = output_accessor.shard_pages(output_shard_id);
        for (const auto& output_page : output_shard_pages) {

            // For each output page, calculate which INPUT page contributes to it
            uint32_t output_page_id = output_page.page_id();

            // Convert output page ID to 3D coordinates (n, d, h, w)
            // For tensor [N, D, H, W, C]: page_id = n*(D*H*W) + d*(H*W) + h*W + w
            uint32_t output_dhw_volume = output_d * output_h * output_w;
            uint32_t n = output_page_id / output_dhw_volume;
            uint32_t remaining = output_page_id % output_dhw_volume;

            uint32_t output_hw_volume = output_h * output_w;
            uint32_t out_d = remaining / output_hw_volume;
            remaining = remaining % output_hw_volume;

            uint32_t out_h = remaining / output_w;
            uint32_t out_w = remaining % output_w;

            // Calculate corresponding INPUT coordinates (nearest neighbor upsampling)
            uint32_t input_d_coord = out_d / scale_factor_d;  // Integer division for nearest neighbor
            uint32_t input_h_coord = out_h / scale_factor_h;
            uint32_t input_w_coord = out_w / scale_factor_w;

            // Calculate input page ID from input coordinates
            uint32_t input_dhw_volume = input_d * input_h * input_w;
            uint32_t input_hw_volume = input_h * input_w;
            uint32_t input_page_id = n * input_dhw_volume +
                                   input_d_coord * input_hw_volume +
                                   input_h_coord * input_w +
                                   input_w_coord;

            // Use TensorAccessor to get input page address
            uint64_t input_noc_addr = input_accessor.get_noc_addr(input_page_id);
            uint64_t output_noc_addr = output_page.get_noc_addr();

            // Copy input stick to output location
            noc_async_read(input_noc_addr, output_noc_addr, stick_nbytes);
        }
    }
    noc_async_read_barrier();
}
```

#### Key Advantages of Output-Perspective Approach:
1. **No config tensor dependency** - uses TensorAccessor directly
2. **Simplified program factory** - only one kernel to manage
3. **Output-driven computation** - work distribution based on output shards (as requested)
4. **Efficient address calculation** - for each output page, calculate which input page to fetch
5. **Work splitting** - NCRISC/BRISC handle different output shards
6. **Direct shard-to-shard transfer** - no intermediate CB needed
7. **Memory-efficient** - only fetches input data that's actually needed

### Phase 4: Integration and Testing
**Objective**: Integrate height sharded support into main operation

#### Integration Points:
1. **Dispatch Logic**: Add height sharded path to main operation dispatch
2. **Memory Config Validation**: Ensure proper memory config propagation
3. **Performance Optimization**: Optimize for height sharded access patterns

#### Testing Strategy:
1. **Run existing interleaved tests** - ensure no regression
2. **Run new height sharded tests** - validate new functionality
3. **Cross-validation** - compare height sharded vs interleaved results
4. **Performance benchmarking** - measure height sharded performance

## Test-Driven Development Approach

### Test Execution Strategy:
1. **Implement one test case at a time**
2. **Run test with 30-second timeout** to catch hanging issues
3. **Stop immediately when first test fails** - debug before continuing
4. **Each test should validate**:
   - Correct output shapes
   - Numerical correctness vs PyTorch reference
   - Memory layout preservation
   - No crashes or hangs

### Test Categories by Priority:
1. **P0 (Critical)**: Basic functionality, simple shard configs
2. **P1 (Important)**: Complex shard configs, edge cases
3. **P2 (Nice-to-have)**: Performance tests, large tensor tests

### Example Test Template:
```python
@pytest.mark.timeout(30)
def test_upsample3d_height_sharded_basic(device):
    """Test basic height sharded upsample3d functionality"""

    # Create input tensor in NDHWC format for TTNN
    input_shape_ncdhw = [1, 32, 2, 8, 8]  # N,C,D,H,W (torch format)
    input_shape_ndhwc = [1, 2, 8, 8, 32]  # N,D,H,W,C (ttnn format)

    # Get device's actual compute grid for realistic testing
    device_grid = device.compute_with_storage_grid_size()

    # Create height sharded memory config using device capabilities
    height_sharded_memory_config = ttnn.create_sharded_memory_config(
        shape=input_shape_ndhwc,
        core_grid=ttnn.CoreGrid(y=min(2, device_grid.y), x=min(2, device_grid.x)),  # Device-aware core grid
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )

    # Create torch input and reference output
    torch.manual_seed(0)
    input_ncdhw = torch.randn(input_shape_ncdhw, dtype=torch.bfloat16)
    input_ndhwc = input_ncdhw.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

    # Create height sharded input tensor
    tt_input_interleaved = ttnn.from_torch(input_ndhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input_height_sharded = ttnn.to_memory_config(tt_input_interleaved, height_sharded_memory_config)

    # Perform upsample3d operation
    scale_factor = (2, 2, 2)
    tt_output = ttnn.upsample3d(tt_input_height_sharded, scale_factor)

    # Validate output is height sharded
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    # Convert to torch and validate correctness
    output_ndhwc = ttnn.to_torch(tt_output)

    # Create torch reference
    torch_upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result_ncdhw = torch_upsample(input_ncdhw)
    torch_result_ndhwc = torch_result_ncdhw.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC

    # Validate shapes and values
    assert list(output_ndhwc.shape) == list(torch_result_ndhwc.shape)
    assert torch.allclose(output_ndhwc, torch_result_ndhwc, atol=1e-2, rtol=1e-2)
```

## Risk Mitigation

### Potential Issues:
1. **Shard boundary alignment** with upsampling factors
2. **Work distribution** complexity for asymmetric scaling
3. **Memory access patterns** efficiency for height sharded data
4. **Test timeouts** due to inefficient implementation

### Mitigation Strategies:
1. **Start with simple cases** (uniform scaling, power-of-2 shards)
2. **Extensive logging** during development
3. **Incremental testing** with immediate failure stops
4. **Reference implementation study** from existing sharded ops

## Success Criteria

### Functional Requirements:
- ✅ All height sharded tests pass
- ✅ No regression in existing interleaved functionality
- ✅ Correct numerical results vs PyTorch reference
- ✅ No memory leaks or crashes
- ✅ Support for various scale factors and shard configurations

### Performance Requirements:
- ✅ Height sharded performance comparable to interleaved
- ✅ Efficient memory access patterns
- ✅ Scalable to multiple cores

### Code Quality Requirements:
- ✅ Clean separation of height sharded vs interleaved paths
- ✅ Proper error handling and validation
- ✅ Comprehensive test coverage
- ✅ Documentation updates

## Implementation Timeline

1. **Phase 1 (Test Setup)**: 2-3 days
2. **Phase 2 (Core Implementation)**: 2-3 days
3. **Phase 3 (Single Kernel Implementation)**: 2-3 days *(Simplified - only one kernel needed!)*
4. **Phase 4 (Integration & Testing)**: 2-3 days

**Total Estimated Duration**: 8-12 days *(Reduced due to single kernel approach)*

---

This plan provides a structured approach to implementing height sharded support for Upsample 3D while maintaining high code quality and avoiding regressions through comprehensive testing.
