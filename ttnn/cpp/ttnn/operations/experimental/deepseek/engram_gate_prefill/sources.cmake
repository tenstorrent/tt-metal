# Source files for ttnn_op_experimental_deepseek_engram_gate_prefill.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_PREFILL_API_HEADERS engram_gate_prefill.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_PREFILL_SRCS
    engram_gate_prefill.cpp
    device/engram_gate_prefill_device_operation.cpp
    device/engram_gate_prefill_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate_prefill/CMakeLists.txt (see the `if(TARGET ttnn)` block).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_PREFILL_NANOBIND_SRCS engram_gate_prefill_nanobind.cpp)
