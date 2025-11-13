# TDD Instructional Prompt for TTNN Operation Development

## Overview
I need you to help implementing a test-driven development plan for implementing the "nearest" (neighbor) sampling mode for grid sampling TTNN operation, looking at how "bilinear" sampling mode is implemented.

**Context To Look At:**
- Example of adding a TTNN device operation https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/adding_new_ttnn_operation.html#example-of-adding-a-new-device-operation
- Existing (bilinear) grid sampling TTNN operation is in ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_op.cpp (above in the call stack are python bindings, and below in the call stack is program factory, kernels etc)
- Skeleton for the nearest-neighbor sampling program factory is in ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_nearest_program_factory.cpp. It doesn't compile, lacks hardware kernels and it is not integrated with the rest of the grid_sample_op.cpp. This is yours to fill-in.

**Approach Requirements:**
1. Use test-driven development with Python-ONLY tests driving implementation
2. Work backwards from Python API to kernel implementation
3. Each stage must have passing tests before proceeding
4. Use DeepWiki for architecture questions instead of searching
5. Create temporary test folder for development iterations
6. Only implement minimal code to pass each test stage

### Key Principles

#### 1. Research Phase
Understand how the existing (bilinear) grid sampling operation is implemented and tested
Start from tests/ttnn/unit_tests/operations/pool/test_grid_sample.py and go down the call stack
(python bindings, op implementation in ttnn/cpp/ttnn/operations/pool/grid_sample, including program factory, kernels etc). Use the example of adding a new TTNN operation from above to better understand the necessary components.

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
The plan should outline gradual implementation of the operation accompanied by tests that verify correctness in this specific order:
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

# Implementation: Minimal Python binding that throws NotImplementedError
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

# Implementation: Use ttnn::register_operation and bind_registered_operation
```

#### Stage 4: Device Operation
```python
# Test: Device operation validates and computes output shape
# Error should be about program creation, not validation

# Implementation: Device op with validate() and compute_output_specs()
```

#### Stage 5: Program Factory
```python
# Test: Program structure created with CBs and work distribution
# Error should mention kernels

# Implementation: Basic program factory with CB creation, throw before kernels
```

#### Stage 6: Kernel Compilation
```python
# Test: Kernels compile without syntax errors
# Should fail at runtime, not compilation

# Implementation: Minimal kernels that compile
```

#### Stage 7: Kernel Correctness
```python
# Test: Multiple test cases with increasing complexity. Edge cases

# Implementation: Complete kernel logic
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

### File Creation Order

1. **Tests First** (test_dev/test_stage1_*.py)
2. **Python Binding** (minimal stub in pybind.cpp)
3. **Host Operation** (operation.hpp/cpp with validation)
4. **Device Operation** (device/operation_op.hpp/cpp)
5. **Program Factory** (device/operation_program_factory.cpp)
6. **Kernels** (kernels/dataflow/*.cpp)

### Success Criteria Checklist

- [ ] All 7 stage tests pass independently
- [ ] No memory leaks (valgrind clean)
- [ ] No Watcher errors
- [ ] Clear error messages at each stage
- [ ] Minimal implementation per stage
- [ ] Documentation in code
- [ ] Build system integration works

### Anti-Patterns to Avoid

1. **Don't implement ahead** - Only code to pass current test
2. **Don't skip stages** - Each builds on previous. DO NOT move forward until a stage is complete.
3. **Don't ignore test failures** - Fix before proceeding
4. **Don't optimize early** - Get it working first
5. **Don't search blindly** - Use DeepWiki for architecture questions

## Summary

This TDD approach ensures:
- **Systematic progress** with clear milestones
- **Early error detection** at the right abstraction level
- **Minimal debugging** through incremental development
- **Confidence** through comprehensive testing
- **Documentation** through test examples

The key insight is to work backwards from the user-facing API to the low-level implementation, with tests driving each stage of development. This prevents over-engineering and ensures each component is properly integrated before moving to the next level of complexity.

## Output
Save the plan (work you should do for all 7 stages) as ttnn/cpp/ttnn/operations/pool/grid_sample/device/grid_sample_nearest_TDD_plan.md
This should be the ONLY artifact of this phase.
