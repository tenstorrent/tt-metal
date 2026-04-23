# Source files for ttnn_op_experimental_unary_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_UNARY_BACKWARD_SRCS
    gelu_backward/device/gelu_backward_device_operation.cpp
    gelu_backward/device/gelu_backward_program_factory.cpp
    gelu_backward/gelu_backward.cpp
)

set(TTNN_OP_EXPERIMENTAL_UNARY_BACKWARD_API_HEADERS gelu_backward/gelu_backward.hpp)
