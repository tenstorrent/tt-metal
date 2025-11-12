# TDD Instructional Prompt for TTNN Operation Development

## Overview
This document provides an optimized instructional prompt for developing new TTNN operations using a test-driven development (TDD) approach. It incorporates lessons learned from implementing the upsample 3D operation and follows the official TTNN operation structure as documented at https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html.

## Instructional Prompt Template

### Initial Request Format
"I need you to help implement a new TTNN operation: [OPERATION_NAME] for [TENSOR_DIMENSIONS]D tensors.

**Context Files Provided:**
- `[operation]_context.MD` - Analysis of existing similar operations
- `[operation]_factoryPlan.MD` - Initial design plan
- `[operation]_program_factory.cpp` - Partially implemented factory (if any)

**Approach Requirements:**
1. Use test-driven development with Python tests driving implementation
2. Follow official TTNN operation structure with nested structs
3. Work backwards from Python API to kernel implementation
4. Each stage must have passing tests before proceeding
5. Use DeepWiki for architecture questions instead of searching
6. Create temporary test folder for development iterations
7. Only implement minimal code to pass each test stage"

### TTNN Operation Structure Requirements

#### Required File Structure
Per TTNN guidelines, every operation must have:
```
ttnn/cpp/ttnn/operations/<category>/<operation>/
├── <operation>.hpp                              # Main operation with Execute struct
├── <operation>.cpp                              # Implementation of invoke()
├── device/
│   ├── <operation>_device_operation.hpp        # Device op with nested structs
│   ├── <operation>_device_operation.cpp        # Device op implementation
│   └── <operation>_program_factory.cpp         # Program factory implementations

ttnn/python/ttnn/operations/<category>/<operation>/
├── <operation>_pybind.hpp                      # Python bindings
└── <operation>_golden.py                       # Golden reference (optional)
```

#### Mandatory Device Operation Structure
Every device operation MUST contain these nested structs:
```cpp
struct OperationDeviceOperation {
    // 1. Operation attributes (non-tensor parameters)
    struct operation_attributes_t {
        // All non-tensor parameters
        MemoryConfig memory_config;
        // ... other parameters
    };

    // 2. Tensor arguments
    struct tensor_args_t {
        const Tensor& input;
        // ... other tensors
    };

    // 3. Program factory (at least one required)
    struct ProgramFactory {
        struct shared_variables_t { /* cached variables */ };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t&,
            const tensor_args_t&,
            tensor_return_value_t<Tensor>&);

        static void override_runtime_arguments(
            cached_program_t&,
            const operation_attributes_t&,
            const tensor_args_t&,
            tensor_return_value_t<Tensor>&);
    };

    // Required static methods
    static program_factory_t select_program_factory(const operation_attributes_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static TensorSpec compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t<Tensor> create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(/* user args */);
};
```

#### Registration Requirements
Operations must be registered using:
```cpp
namespace ttnn {
    constexpr auto operation_name = ttnn::register_operation<
        "ttnn::operation_name",
        ttnn::operations::category::ExecuteOperation>();
}
```

Python bindings must use:
```cpp
ttnn::bind_registered_operation(
    module,
    ttnn::operation_name,
    "Documentation string",
    ttnn::pybind_overload_t{ /* overloads */ });
```

### Key Principles

#### 1. Research Phase - Use DeepWiki First
Instead of brute-force searching through the repository, use DeepWiki for high-level questions:
```
Ask DeepWiki about:
- Complete implementation stack for similar operations
- Kernel implementation patterns (reader/writer/compute)
- Testing patterns and progression
- Memory layout handling patterns
- Validation and error handling patterns
```

#### 2. Test-Driven Development Stages (Reverse Order)
Tests should verify in this specific order:
1. **API Existence** - Python binding exists and is callable
2. **Parameter Validation** - Input validation works correctly
3. **TTNN Registration** - Operation is properly registered
4. **Device Operation** - Validates inputs and computes output shape
5. **Program Factory** - Creates program with work distribution
6. **Kernel Compilation** - Kernels compile without errors
7. **Kernel Correctness** - Kernels execute with correct results
8. **Final Integration** - Comprehensive tests (only after stages 1-7 pass)

#### 3. Test Organization
```
ttnn/cpp/ttnn/operations/[category]/[operation]/test_dev/
├── test_stage1_api_exists.py
├── test_stage2_validation.py
├── test_stage3_registration.py
├── test_stage4_device_op.py
├── test_stage5_program_factory.py
├── test_stage6_kernel_compile.py
└── test_stage7_kernel_correct.py
```

#### 4. Implementation Strategy
- **Minimal Implementation**: Write only enough code to pass current test
- **Fail First**: Test must fail before implementation
- **Incremental Progress**: Never skip stages
- **Clear Errors**: Each stage should fail with meaningful errors

### Specific Instructions for Each Stage

#### Stage 1: API Existence
```python
# Test: API is accessible
assert hasattr(ttnn, 'operation_name')
assert callable(ttnn.operation_name)

# Implementation:
# 1. Create main operation header with ExecuteOperation struct
# 2. Register with ttnn::register_operation
# 3. Create Python binding with ttnn::bind_registered_operation
```

#### Stage 2: Parameter Validation
```python
# Test: Validates tensor rank, layout, data types, parameter ranges
with pytest.raises(RuntimeError) as exc_info:
    ttnn.operation(invalid_input)
assert "meaningful error" in str(exc_info.value)

# Implementation: Host operation with TT_FATAL validations
```

#### Stage 3: TTNN Registration
```python
# Test: Operation has correct signature and documentation
sig = inspect.signature(ttnn.operation)
assert 'expected_params' in sig.parameters

# Implementation: Verify registration and binding are complete
```

#### Stage 4: Device Operation
```python
# Test: Device operation validates and computes output shape
# Error should be about program creation, not validation

# Implementation: Create device operation with:
# - operation_attributes_t struct
# - tensor_args_t struct
# - validate_on_program_cache_miss()
# - validate_on_program_cache_hit()
# - compute_output_specs()
# - create_output_tensors()
# - invoke() to map user args
# - select_program_factory()
```

#### Stage 5: Program Factory
```python
# Test: Program structure created with CBs and work distribution
# Error should mention kernels

# Implementation: Create program factory with:
# - shared_variables_t struct for caching
# - create() method that builds Program
# - override_runtime_arguments() for buffer updates
# - Circular buffer creation
# - Work distribution across cores
```

#### Stage 6: Kernel Compilation
```python
# Test: Kernels compile without syntax errors
# Should fail at runtime, not compilation

# Implementation: Minimal kernels that compile
```

#### Stage 7: Kernel Correctness
```python
# Test: Multiple test cases with increasing complexity:
# - Identity operation (scale=1)
# - Single dimension scaling
# - Uniform scaling
# - Mixed scaling
# - Edge cases

# Implementation: Complete kernel logic
```

#### Stage 8: Golden Function (Optional but Recommended)
```python
# Test: Golden function attached and validates against PyTorch
assert hasattr(ttnn.operation, 'golden_function')
torch.testing.assert_close(ttnn_output, golden_output)

# Implementation:
# 1. Create golden function using PyTorch
# 2. Use ttnn.attach_golden_function()
# 3. Add preprocess/postprocess if needed
```

### Common Patterns to Follow

#### Memory Layout Handling
```cpp
// Support interleaved first
TT_FATAL(input.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
         "Only INTERLEAVED memory layout is supported");

// Add sharded support later as optimization
```

#### Work Distribution
```cpp
// Use standard pattern
const auto [num_cores, all_cores, core_group_1, core_group_2,
            work_per_core_group_1, work_per_core_group_2] =
    split_work_to_cores(compute_grid_size, work_units_to_split);
```

#### Circular Buffer Pattern
```cpp
// Standard CB indices
uint32_t cb_index = tt::CBIndex::c_0;  // Input
uint32_t cb_output = tt::CBIndex::c_16; // Output (if needed)

// Double buffer if processing multiple units
uint32_t num_pages = work_per_core > 1 ? 2 : 1;
```

### Testing Best Practices

#### Progressive Complexity
1. Start with smallest possible tensors
2. Test identity operations first (no transformation)
3. Test one dimension at a time
4. Combine dimensions gradually
5. Add edge cases last

#### Debugging Support
```bash
# Environment variables for debugging
export TT_METAL_WATCHER=10
export TT_METAL_DPRINT_CORES=(0,0)
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}'
```

#### Test Fixtures
```python
@pytest.fixture
def device():
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)
```

### Error Messages to Expect

| Stage | Expected Error Contains |
|-------|------------------------|
| 1 | "not implemented" or "attribute" |
| 2 | Specific validation message |
| 3 | "operation" or "device" |
| 4 | "program" or "create" |
| 5 | "kernel" or "buffer" |
| 6 | No compilation errors |
| 7 | No errors (tests pass) |

### File Creation Order (Following TTNN Structure)

1. **Tests First** (test_dev/test_stage1_*.py)
2. **Main Operation Header** (operation.hpp with ExecuteOperation struct)
3. **Python Binding** (python/ttnn/operations/category/operation_pybind.hpp)
4. **Main Operation Implementation** (operation.cpp with invoke())
5. **Device Operation Header** (device/operation_device_operation.hpp with nested structs)
6. **Device Operation Implementation** (device/operation_device_operation.cpp)
7. **Program Factory** (device/operation_program_factory.cpp with create/override methods)
8. **Kernels** (kernels/dataflow/*.cpp)
9. **Golden Function** (python/ttnn/operations/category/operation_golden.py)

### Success Criteria Checklist

- [ ] All 8 stage tests pass independently
- [ ] Follows TTNN operation structure with nested structs
- [ ] Proper registration with ttnn::register_operation
- [ ] Python bindings use ttnn::bind_registered_operation
- [ ] Program factory has create/override_runtime_arguments methods
- [ ] Golden function attached (if Stage 8 implemented)
- [ ] No memory leaks (valgrind clean)
- [ ] No Watcher errors
- [ ] Clear error messages at each stage
- [ ] Minimal implementation per stage
- [ ] Documentation in code
- [ ] Build system integration works

### Anti-Patterns to Avoid

1. **Don't implement ahead** - Only code to pass current test
2. **Don't skip stages** - Each builds on previous
3. **Don't ignore test failures** - Fix before proceeding
4. **Don't optimize early** - Get it working first
5. **Don't search blindly** - Use DeepWiki for architecture questions
6. **Don't create final tests early** - Wait until stage 7 passes
7. **Don't skip the nested struct pattern** - TTNN requires specific structure
8. **Don't use raw operation::run** - Use ttnn::device_operation::run
9. **Don't forget program caching** - Implement shared_variables_t properly
10. **Don't mix validation** - Separate cache miss vs cache hit validation

### Example Usage

```markdown
"I need to implement a 3D convolution operation for 5D tensors (N, D, H, W, C).

I have created:
- conv3d_context.MD analyzing conv2d implementation
- conv3d_factoryPlan.MD with design approach
- Partial conv3d_program_factory.cpp

Please help me create a TDD plan following the 7-stage approach, starting with Python API tests and working down to kernels. Tests should be in test_dev/ folder and we should verify each stage passes before proceeding."
```

### Final Integration (Stage 8)

Only after stages 1-7 pass:
1. Move to final test location: `tests/ttnn/unit_tests/operations/`
2. Create comprehensive parametrized tests
3. Add performance benchmarks
4. Test with real model scenarios
5. Document usage examples

### Performance Optimization (Post-TDD)

After correctness is established:
1. Profile with Tracy
2. Optimize NOC transfers
3. Add sharded memory support
4. Implement tiled layout path
5. Vectorize where possible

## Summary

This TDD approach ensures:
- **TTNN Compliance** with official operation structure requirements
- **Systematic progress** with clear milestones
- **Early error detection** at the right abstraction level
- **Minimal debugging** through incremental development
- **Confidence** through comprehensive testing
- **Documentation** through test examples
- **Program caching** support through proper factory structure
- **Golden function** validation against PyTorch reference

The key insight is to work backwards from the user-facing API to the low-level implementation, with tests driving each stage of development while strictly following the TTNN operation structure. This prevents over-engineering, ensures proper integration with the TTNN framework, and guarantees that each component is properly tested before moving to the next level of complexity.

### Critical TTNN Requirements Summary

1. **Nested Structs**: Device operations MUST have `operation_attributes_t`, `tensor_args_t`, and factory structs
2. **Factory Methods**: Program factories MUST implement `create()` and `override_runtime_arguments()`
3. **Registration**: Operations MUST use `ttnn::register_operation` and `ttnn::bind_registered_operation`
4. **Validation Split**: Separate validation for cache miss (full) and cache hit (light)
5. **Return Types**: `compute_output_specs()` and `create_output_tensors()` must have parallel return types
6. **Golden Function**: Use `ttnn.attach_golden_function()` for PyTorch reference validation

Following this structure ensures compatibility with TTNN's program caching, validation pipeline, and device operation framework.
