# Source files for ttnn_op_experimental_conv1d_depthwise.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_CONV1D_DEPTHWISE_API_HEADERS
    conv1d_depthwise.hpp
    device/conv1d_depthwise_device_operation.hpp
)

set(TTNN_OP_EXPERIMENTAL_CONV1D_DEPTHWISE_SRCS
    conv1d_depthwise.cpp
    device/conv1d_depthwise_device_operation.cpp
    device/conv1d_depthwise_program_factory.cpp
)
