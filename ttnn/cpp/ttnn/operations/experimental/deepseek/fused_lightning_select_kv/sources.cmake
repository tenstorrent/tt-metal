# Source files for ttnn_op_experimental_deepseek_fused_lightning_select_kv.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_FUSED_LIGHTNING_SELECT_KV_API_HEADERS fused_lightning_select_kv.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_FUSED_LIGHTNING_SELECT_KV_SRCS
    fused_lightning_select_kv.cpp
    device/fused_lightning_select_kv_device_operation.cpp
    device/fused_lightning_select_kv_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/fused_lightning_select_kv/CMakeLists.txt
# (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_FUSED_LIGHTNING_SELECT_KV_NANOBIND_SRCS fused_lightning_select_kv_nanobind.cpp)
