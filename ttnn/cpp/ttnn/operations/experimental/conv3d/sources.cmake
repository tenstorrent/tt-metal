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

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/conv3d/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_CONV3D_NANOBIND_SRCS conv3d_nanobind.cpp)
