# Source files for this op's Python bindings.
# Module owners should update this file when adding/removing/renaming source files.

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_GENERALIZED_MOE_GATE_NANOBIND_SRCS generalized_moe_gate_nanobind.cpp)
