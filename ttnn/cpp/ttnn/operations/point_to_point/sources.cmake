# Source files for ttnn_op_point_to_point.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_POINT_TO_POINT_API_HEADERS point_to_point.hpp)

set(TTNN_OP_POINT_TO_POINT_SRCS
    device/host/point_to_point_device_op.cpp
    device/host/send_program_factory.cpp
    device/host/receive_program_factory.cpp
    device/host/local_copy_program_factory.cpp
    point_to_point.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/point_to_point/CMakeLists.txt (see the
# `if(TARGET ttnn)` block there). Listed here rather than inline in
# CMakeLists.txt so that add/remove/rename doesn't touch a file with
# metalium-developers-infra as a required co-owner.
set(TTNN_OP_POINT_TO_POINT_NANOBIND_SRCS point_to_point_nanobind.cpp)
