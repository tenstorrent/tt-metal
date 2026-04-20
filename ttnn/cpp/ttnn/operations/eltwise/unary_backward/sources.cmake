# Source files for ttnn_op_eltwise_unary_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_UNARY_BACKWARD_SRCS
    unary_backward.cpp
    tanh_bw/device/tanh_bw_device_operation.cpp
    tanh_bw/device/tanh_bw_program_factory.cpp
)

set(TTNN_OP_ELTWISE_UNARY_BACKWARD_API_HEADERS unary_backward.hpp)
