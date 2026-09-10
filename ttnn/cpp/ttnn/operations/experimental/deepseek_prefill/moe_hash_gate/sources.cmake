set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_HASH_GATE_API_HEADERS moe_hash_gate.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_HASH_GATE_SRCS
    device/moe_hash_gate_device_operation.cpp
    device/moe_hash_gate_program_factory.cpp
    moe_hash_gate.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_HASH_GATE_NANOBIND_SRCS moe_hash_gate_nanobind.cpp)
