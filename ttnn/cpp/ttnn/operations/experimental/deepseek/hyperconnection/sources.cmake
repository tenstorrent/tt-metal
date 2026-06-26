# Source files for ttnn_op_experimental_deepseek_hyperconnection.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_HYPERCONNECTION_API_HEADERS fused_hyperconnection.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_HYPERCONNECTION_SRCS
    fused_hyperconnection.cpp
    device/fused_pre_post_device_operation.cpp
    device/fused_pre_post_program_factory.cpp
    device/sinkhorn_device_operation.cpp
    device/sinkhorn_program_factory.cpp
)
