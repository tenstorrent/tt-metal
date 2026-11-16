# Source files for ttnn_op_experimental_deepseek_prefill_moe_fused_swiglu.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_FUSED_SWIGLU_API_HEADERS moe_fused_swiglu.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_FUSED_SWIGLU_NANOBIND_SRCS moe_fused_swiglu_nanobind.cpp)
