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
    device/hybrid_overlap_program_factory.cpp
    device/hybrid_program_factory.cpp
    hybrid_routed_expert_ffn.cpp
    # combine_fabric2d, carried the same way (device/combine, device/kernels/combine) under the
    # hybrid_routed_expert_ffn::combine and hyb_cmbf2d namespaces, so the overlap can change it
    # without touching the standalone op.
    device/combine/combine_fabric2d_assignments.cpp
    device/combine/combine_fabric2d_placement.cpp
    device/combine/combine_fabric2d_program_factory.cpp
    device/combine/combine_fabric2d_device_operation.cpp
)

# Registered on the shared `ttnn` Python module target from this op's CMakeLists.txt (see the
# `if(TARGET ttnn)` block there). Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HYBRID_ROUTED_EXPERT_FFN_NANOBIND_SRCS hybrid_routed_expert_ffn_nanobind.cpp)
