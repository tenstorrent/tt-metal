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

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_HYPERCONNECTION_NANOBIND_SRCS fused_hyperconnection_nanobind.cpp)
