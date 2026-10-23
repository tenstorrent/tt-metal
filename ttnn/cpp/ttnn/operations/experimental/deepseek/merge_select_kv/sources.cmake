# Source files for ttnn_op_experimental_deepseek_merge_select_kv.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MERGE_SELECT_KV_API_HEADERS merge_select_kv.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MERGE_SELECT_KV_SRCS
    merge_select_kv.cpp
    device/merge_select_kv_device_operation.cpp
    device/merge_select_kv_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/merge_select_kv/CMakeLists.txt
# (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MERGE_SELECT_KV_NANOBIND_SRCS merge_select_kv_nanobind.cpp)
