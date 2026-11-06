# Source files for ttnn_op_experimental_deepseek_prefill_unified_routed_expert_combine.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_API_HEADERS
    unified_routed_expert_combine.hpp
    device/unified_routed_expert_combine_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_SRCS
    device/unified_routed_expert_combine_device_operation.cpp
    device/unified_routed_expert_combine_program_factory.cpp
    unified_routed_expert_combine.cpp
)

# Registered on the shared `ttnn` Python module target from this op's CMakeLists.txt (see the
# `if(TARGET ttnn)` block there). Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_COMBINE_NANOBIND_SRCS
    unified_routed_expert_combine_nanobind.cpp
)
