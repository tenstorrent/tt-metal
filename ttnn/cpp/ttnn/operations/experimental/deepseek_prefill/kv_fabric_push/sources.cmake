# Source files for ttnn_op_experimental_deepseek_prefill_kv_fabric_push.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_KV_FABRIC_PUSH_SRCS
    device/kv_fabric_push_device_operation.cpp
    device/kv_fabric_push_program_factory.cpp
    kv_fabric_push.cpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_KV_FABRIC_PUSH_API_HEADERS kv_fabric_push.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/kv_fabric_push/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_KV_FABRIC_PUSH_NANOBIND_SRCS kv_fabric_push_nanobind.cpp)
