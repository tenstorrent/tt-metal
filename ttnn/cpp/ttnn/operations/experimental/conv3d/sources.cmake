# Source files for ttnn_op_experimental_conv3d.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_CONV3D_API_HEADERS
    conv3d.hpp
    prepare_conv3d_weights.hpp
    device/conv3d_device_operation.hpp
    device/conv3d_device_operation_types.hpp
    device/conv3d_program_factory.hpp
)

set(TTNN_OP_EXPERIMENTAL_CONV3D_SRCS
    conv3d.cpp
    prepare_conv3d_weights.cpp
    device/conv3d_device_operation.cpp
    device/conv3d_program_factory.cpp
)
