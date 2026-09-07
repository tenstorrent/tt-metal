# Source files for ttnn_op_experimental_deepseek_prefill_dispatch_fabric2d.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_FABRIC2D_API_HEADERS dispatch_fabric2d.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_FABRIC2D_SRCS
    device/dispatch_fabric2d_device_operation.cpp
    device/dispatch_fabric2d_assignments.cpp
    device/dispatch_fabric2d_placement.cpp
    device/dispatch_fabric2d_program_factory.cpp
    dispatch_fabric2d.cpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_FABRIC2D_NANOBIND_SRCS dispatch_fabric2d_nanobind.cpp)
