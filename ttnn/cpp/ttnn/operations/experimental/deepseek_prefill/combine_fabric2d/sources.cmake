# Source files for ttnn_op_experimental_deepseek_prefill_combine_fabric2d.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_FABRIC2D_API_HEADERS combine_fabric2d.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_FABRIC2D_SRCS
    device/combine_fabric2d_device_operation.cpp
    device/combine_fabric2d_program_factory.cpp
    combine_fabric2d.cpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_FABRIC2D_NANOBIND_SRCS combine_fabric2d_nanobind.cpp)
