# Source files for ttnn_op_experimental_deepseek_moe_fused_experts.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_EXPERTS_API_HEADERS
    fused_experts.hpp
    persistent_moe_request.hpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_EXPERTS_SRCS
    fused_experts.cpp
    persistent_moe_request.cpp
    device/d2d_routes.cpp
    device/fused_experts_device_operation.cpp
    device/fused_experts_program_factory.cpp
    device/persistent_moe_request_device_operation.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/moe/fused_experts/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_FUSED_EXPERTS_NANOBIND_SRCS fused_experts_nanobind.cpp)
