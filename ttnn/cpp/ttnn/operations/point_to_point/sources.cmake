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
