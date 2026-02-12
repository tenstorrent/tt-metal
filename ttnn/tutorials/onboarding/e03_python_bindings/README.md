# E03: Python Bindings

Create a ttnn operation with Python bindings.

## Goal

Learn the operation registration and Python binding pattern:
- Define operation struct with `invoke()`
- Register with `ttnn::register_operation`
- Bind to Python with nanobind

## Prerequisites

Study:
- `ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp`

## Structure

```
e03_python_bindings/
├── exercise.py              # Calls your operation
├── solution.py              # Reference
├── reference.py             # PyTorch reference
├── test.py                  # Pytest
├── exercise_cpp/            # Your implementation (stubs)
│   ├── eltwise_add.hpp      # Operation struct
│   ├── eltwise_add.cpp      # invoke() calls ttnn::add
│   └── eltwise_add_nanobind.cpp
└── solution_cpp/            # Complete implementation
```

## Exercise

1. Implement `eltwise_add.hpp` - define struct with `invoke()`
2. Implement `eltwise_add.cpp` - call `ttnn::add(a, b)`
3. Implement `eltwise_add_nanobind.cpp` - bind to Python

## Build & Test

From project root (`tt-metal/`):

```bash
# Build onboarding module (after tt-metal is built)
cmake --build build -- onboarding

# Run tests
ttnn/tutorials/onboarding/run.sh "e03 and solution"   # Should pass
ttnn/tutorials/onboarding/run.sh "e03 and exercise"   # Your implementation
```
