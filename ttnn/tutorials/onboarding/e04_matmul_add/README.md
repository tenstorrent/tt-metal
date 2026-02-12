# E04: Matmul + Add Kernels

Write custom kernels for matmul + add.

## Goal

Learn the full device operation pattern with custom kernels:
- Create device operation structure (operation_attributes_t, tensor_args_t)
- Implement ProgramFactory with custom kernels
- Write reader, compute, and writer kernels
- Compute: output = a @ b + c

## Prerequisites

- Complete e03 first
- Study: `tt_metal/programming_examples/matmul/matmul_single_core/`
- Study: `ttnn/cpp/ttnn/operations/full/`

## Structure

```
e04_matmul_add/
├── exercise.py              # Calls your operation
├── solution.py              # Reference
├── reference.py             # PyTorch reference
├── test.py                  # Pytest
├── exercise_cpp/            # Your implementation (stubs)
│   ├── matmul_add.hpp
│   └── device/
│       ├── matmul_add_device_operation.hpp
│       └── kernels/         # Your kernels
└── solution_cpp/            # Complete implementation
    ├── matmul_add.hpp, cpp
    ├── matmul_add_nanobind.cpp
    └── device/
        ├── matmul_add_device_operation.*
        ├── matmul_add_program_factory.cpp
        └── kernels/         # Custom kernels
```

## Exercise

1. Study `ttnn/cpp/ttnn/operations/full/`
2. Implement device operation structure
3. Implement ProgramFactory
4. Write custom kernels in device/kernels/

## Build & Test

From project root (`tt-metal/`):

```bash
# Build onboarding module (after tt-metal is built)
cmake --build build -- onboarding

# Run tests
ttnn/tutorials/onboarding/run.sh "e04 and solution"   # Should pass
ttnn/tutorials/onboarding/run.sh "e04 and exercise"   # Your implementation
```
