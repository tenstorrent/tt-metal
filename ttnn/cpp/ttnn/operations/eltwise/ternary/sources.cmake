# Source files for ttnn_op_eltwise_ternary.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_TERNARY_SRCS
    ternary_composite_op.cpp
    ternary.cpp
    device/ternary_device_operation.cpp
    device/ternary_program_factory.cpp
    device/ternary_op_utils.cpp
)

set(TTNN_OP_ELTWISE_TERNARY_API_HEADERS
    common/ternary_op_types.hpp
    ternary.hpp
)
