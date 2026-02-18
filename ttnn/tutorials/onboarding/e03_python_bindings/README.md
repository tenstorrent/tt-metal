# E03: Python Bindings

Create a ttnn operation with Python bindings.

## Goal

Learn the operation registration and Python binding pattern:
- Define operation struct with `invoke()`
- Register with `ttnn::register_operation`
- Bind to Python with nanobind

## Reference

- `ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp`
- `ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp`

## Key Concepts

### Operation Registration
- Define a struct with a static `invoke()` method
- Register using `ttnn::register_operation<"name", Struct>()`
- Operation becomes callable from C++ and Python

### Python Bindings with Nanobind
- Use `ttnn::bind_registered_operation` to expose to Python
- Specify argument names with `nb::arg()`
- Add docstring for documentation

### Compile-time vs Runtime Arguments
- **Compile-time**: Known when kernel is compiled (dimensions, types)
- **Runtime**: Can change between invocations (buffer addresses)

## Common Pitfalls

1. **Forgetting to register** - Operation must be registered to be callable
2. **Nanobind arg mismatch** - Args in binding must match invoke() signature
3. **Module naming** - Module name must match CMake target name

## Build & Test

```bash
cmake --build build -- onboarding
ttnn/tutorials/onboarding/run.sh "e03 and solution"
ttnn/tutorials/onboarding/run.sh "e03 and exercise"
```
