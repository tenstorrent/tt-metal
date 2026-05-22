# Source files for ttnn_op_conv.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_CONV_SRCS
    conv2d/conv2d.cpp
    conv2d/conv2d_utils.cpp
    conv2d/conv2d_op_program_factory_common.cpp
    conv2d/device/conv2d_device_operation.cpp
    conv2d/device/conv2d_op_sharded_program_factory.cpp
    conv2d/device/conv2d_op_width_sharded_program_factory.cpp
    conv_transpose2d/conv_transpose2d.cpp
    conv_transpose2d/prepare_conv_transpose2d_weights.cpp
    conv2d/prepare_conv2d_weights.cpp
    conv1d/conv1d.cpp
)

set(TTNN_OP_CONV_API_HEADERS
    conv2d/conv2d.hpp
    conv2d/device/conv2d_device_operation_types.hpp
    conv2d/device/conv2d_device_operation.hpp
    conv2d/device/conv2d_op_sharded_program_factory.hpp
    conv2d/device/conv2d_op_width_sharded_program_factory.hpp
)
