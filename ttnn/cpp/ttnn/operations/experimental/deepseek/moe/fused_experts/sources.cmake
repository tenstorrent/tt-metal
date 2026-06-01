# Source files for ttnn_op_experimental_deepseek_moe_fused_experts.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_EXPERTS_API_HEADERS fused_experts.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_EXPERTS_SRCS
    fused_experts.cpp
    device/fused_experts_device_operation.cpp
    device/fused_experts_program_factory.cpp
)
