# Source files for ttnn_op_experimental_rgb_to_yuv.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_RGB_TO_YUV_SRCS
    device/rgb_to_yuv_device_op.cpp
    device/rgb_to_yuv_program_factory.cpp
    rgb_to_yuv.cpp
)

set(TTNN_OP_EXPERIMENTAL_RGB_TO_YUV_API_HEADERS
    rgb_to_yuv.hpp
    device/rgb_to_yuv_device_op_types.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/rgb_to_yuv/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_RGB_TO_YUV_NANOBIND_SRCS rgb_to_yuv_nanobind.cpp)
