# Source files for ttnn_op_experimental_cnn.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_CNN_SRCS
    convert_to_chw/convert_to_chw.cpp
    convert_to_chw/device/convert_to_chw_device_operation.cpp
    convert_to_chw/device/convert_to_chw_program_factory.cpp
    convert_to_hwc/convert_to_hwc.cpp
    convert_to_hwc/device/convert_to_hwc_device_operation.cpp
    convert_to_hwc/device/convert_to_hwc_program_factory.cpp
    convert_to_hwc/device/gather.cpp
)
