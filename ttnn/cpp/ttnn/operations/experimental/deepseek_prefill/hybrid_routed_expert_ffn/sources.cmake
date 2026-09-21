# Source files for ttnn_op_experimental_deepseek_prefill_hybrid_routed_expert_ffn.
# Module owners should update this file when adding/removing/renaming source files.

# The three carried implementations keep the filenames they have upstream so a fix to any of
# moe_fused_swiglu, unified_routed_expert_ffn or combine_fabric2d stays a readable diff against
# its origin. Their host code is renamed into nested namespaces (::fused, ::unified, ::combine,
# and hyb_cmbf2d) because all four ops link into one libttnn.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_API_HEADERS
    hybrid_routed_expert_ffn.hpp
    device/hybrid_routed_expert_ffn_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_SRCS
    device/combine_fabric2d_assignments.cpp
    device/combine_fabric2d_placement.cpp
    device/combine_fabric2d_program_factory.cpp
    device/hybrid_half_merge.cpp
    device/hybrid_routed_expert_ffn_device_operation.cpp
    device/hybrid_program_factory.cpp
    hybrid_routed_expert_ffn.cpp
)

# Registered on the shared `ttnn` Python module target from this op's CMakeLists.txt (see the
# `if(TARGET ttnn)` block there). Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_NANOBIND_SRCS hybrid_routed_expert_ffn_nanobind.cpp)
