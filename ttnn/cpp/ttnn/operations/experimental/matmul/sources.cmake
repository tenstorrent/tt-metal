# Source files for ttnn_op_experimental_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_MATMUL_SRCS
    attn_matmul/attn_matmul.cpp
    attn_matmul/device/attn_matmul_device_operation.cpp
    attn_matmul/device/attn_matmul_program_factory.cpp
    group_attn_matmul/device/group_attn_matmul_device_operation.cpp
    group_attn_matmul/device/group_attn_matmul_program_factory.cpp
    group_attn_matmul/group_attn_matmul.cpp
)
