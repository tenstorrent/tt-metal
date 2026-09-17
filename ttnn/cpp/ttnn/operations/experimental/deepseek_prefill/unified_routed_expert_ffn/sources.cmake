# Source files for ttnn_op_experimental_deepseek_prefill_unified_routed_expert_ffn.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_FFN_API_HEADERS unified_routed_expert_ffn.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_UNIFIED_ROUTED_EXPERT_FFN_NANOBIND_SRCS
    unified_routed_expert_ffn_nanobind.cpp
)
