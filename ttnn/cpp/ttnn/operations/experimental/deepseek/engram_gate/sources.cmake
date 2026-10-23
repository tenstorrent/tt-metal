# Source files for ttnn_op_experimental_deepseek_engram_gate.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_API_HEADERS engram_gate.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_SRCS
    engram_gate.cpp
    device/engram_gate_device_operation.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_NANOBIND_SRCS engram_gate_nanobind.cpp)
