# Source files for ttnn_op_experimental_deepseek_prefill_moe_ag.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_AG_API_HEADERS moe_ag.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_AG_SRCS
    device/moe_ag_common.cpp
    device/moe_ag_local_reduce_device_operation.cpp
    device/moe_ag_local_reduce_program_factory.cpp
    device/moe_ag_route_plan_device_operation.cpp
    device/moe_ag_route_plan_program_factory.cpp
    device/moe_ag_row_ops_device_operation.cpp
    device/moe_ag_row_ops_program_factory.cpp
    moe_ag.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_ag/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_AG_NANOBIND_SRCS moe_ag_nanobind.cpp)
