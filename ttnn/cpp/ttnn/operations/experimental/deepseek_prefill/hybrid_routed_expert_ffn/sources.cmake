# Source files for ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn.
# Module owners should update this file when adding/removing/renaming source files.

# The two carried implementations keep the filenames they have upstream so a fix to either
# moe_fused_swiglu or unified_routed_expert_ffn stays a readable diff against its origin.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_API_HEADERS
    hybrid_routed_expert_ffn.hpp
    device/hybrid_routed_expert_ffn_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_SRCS
    device/hybrid_half_merge.cpp
    device/hybrid_routed_expert_ffn_device_operation.cpp
    device/moe_fused_swiglu_geometry.cpp
    device/hybrid_program_factory.cpp
    hybrid_routed_expert_ffn.cpp
)

# Registered on the shared `ttnn` Python module target from this op's CMakeLists.txt (see the
# `if(TARGET ttnn)` block there). Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_NANOBIND_SRCS hybrid_routed_expert_ffn_nanobind.cpp)
