# Source files for ttnn_op_eltwise_unary.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_UNARY_SRCS
    common/unary_op_utils.cpp
    common/unary_utils.cpp
    device/unary_composite_op.cpp
    device/unary_device_operation.cpp
    device/unary_program_factory.cpp
    unary.cpp
)

set(TTNN_OP_ELTWISE_UNARY_API_HEADERS
    common/unary_op_types.hpp
    common/unary_op_utils.hpp
    device/unary_composite_op.hpp
    unary.hpp
    unary_composite.hpp
)
