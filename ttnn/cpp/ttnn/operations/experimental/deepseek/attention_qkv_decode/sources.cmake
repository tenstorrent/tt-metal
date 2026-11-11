# Source files for ttnn_op_experimental_deepseek_attention_qkv_decode.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ATTENTION_QKV_DECODE_API_HEADERS attention_qkv_decode.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ATTENTION_QKV_DECODE_SRCS
    attention_qkv_decode.cpp
    device/attention_qkv_decode_device_operation.cpp
    device/attention_qkv_decode_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/attention_qkv_decode/CMakeLists.txt
# (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ATTENTION_QKV_DECODE_NANOBIND_SRCS attention_qkv_decode_nanobind.cpp)
