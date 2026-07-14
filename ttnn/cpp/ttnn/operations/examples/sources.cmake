# Source files for ttnn_op_examples.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXAMPLES_SRCS
    example/example.cpp
    example/device/example_device_operation.cpp
    example/device/multi_core_program_factory.cpp
    example/device/single_core_program_factory.cpp
    example_multiple_return/device/example_multiple_return_device_operation.cpp
    example_multiple_return/device/single_core_program_factory.cpp
    example_multiple_return/example_multiple_return.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/examples/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXAMPLES_NANOBIND_SRCS
    example/example_nanobind.cpp
    example_multiple_return/example_multiple_return_nanobind.cpp
    examples_nanobind.cpp
)
