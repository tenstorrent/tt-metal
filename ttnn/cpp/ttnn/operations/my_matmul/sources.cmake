# Source files for ttnn_op_my_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_MY_MATMUL_API_HEADERS my_matmul.hpp)

set(TTNN_OP_MY_MATMUL_SRCS
    device/my_matmul_device_operation.cpp
    device/single_core_program_factory.cpp
    device/multi_core_program_factory.cpp
    my_matmul.cpp
)
