# Source files for ttnn_op_experimental_cnn.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_CNN_API_HEADERS
    convert_to_chw/convert_to_chw.hpp
    convert_to_hwc/convert_to_hwc.hpp
)

set(TTNN_OP_EXPERIMENTAL_CNN_SRCS
    convert_to_chw/convert_to_chw.cpp
    convert_to_chw/device/convert_to_chw_device_operation.cpp
    convert_to_chw/device/convert_to_chw_program_factory.cpp
    convert_to_hwc/convert_to_hwc.cpp
    convert_to_hwc/device/convert_to_hwc_device_operation.cpp
    convert_to_hwc/device/convert_to_hwc_program_factory.cpp
    convert_to_hwc/device/gather.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/cnn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_CNN_NANOBIND_SRCS
    convert_to_chw/convert_to_chw_nanobind.cpp
    convert_to_hwc/convert_to_hwc_nanobind.cpp
)
