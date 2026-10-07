# Source files for ttnn_op_experimental_deepseek_prefill_flat_routed_expert.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_FLAT_ROUTED_EXPERT_API_HEADERS
    flat_routed_expert.hpp
    device/flat_routed_expert_plan.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/flat_routed_expert/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_FLAT_ROUTED_EXPERT_NANOBIND_SRCS flat_routed_expert_nanobind.cpp)
