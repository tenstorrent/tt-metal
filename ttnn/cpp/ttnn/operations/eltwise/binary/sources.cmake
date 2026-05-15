# Source files for ttnn_op_eltwise_binary.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_BINARY_SRCS
    binary.cpp
    common/binary_op_utils.cpp
    device/binary_composite_op.cpp
)

set(TTNN_OP_ELTWISE_BINARY_API_HEADERS
    binary.hpp
    binary_composite.hpp
    common/binary_op_types.hpp
    common/binary_op_utils.hpp
)
