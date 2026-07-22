# Source files for ttnn_op_experimental_yuv_conversion.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_YUV_CONVERSION_SRCS
    device/yuv_conversion_device_op.cpp
    device/yuv_conversion_program_factory.cpp
    yuv_conversion.cpp
)

set(TTNN_OP_EXPERIMENTAL_YUV_CONVERSION_API_HEADERS yuv_conversion.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/yuv_conversion/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_YUV_CONVERSION_NANOBIND_SRCS yuv_conversion_nanobind.cpp)
