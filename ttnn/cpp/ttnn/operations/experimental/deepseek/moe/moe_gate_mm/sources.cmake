# Source files for ttnn_op_experimental_deepseek_moe_moe_gate_mm.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_MOE_GATE_MM_API_HEADERS moe_gate_mm.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_MOE_GATE_MM_SRCS
    device/moe_gate_mm_device_operation.cpp
    device/moe_gate_mm_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_MOE_GATE_MM_NANOBIND_SRCS moe_gate_mm_nanobind.cpp)
