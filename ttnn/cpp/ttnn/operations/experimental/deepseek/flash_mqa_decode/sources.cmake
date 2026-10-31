# Source files for ttnn_op_experimental_deepseek_flash_mqa_decode.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_FLASH_MQA_DECODE_API_HEADERS flash_mqa_decode.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_FLASH_MQA_DECODE_SRCS
    flash_mqa_decode.cpp
    device/flash_mqa_decode_device_operation.cpp
    device/flash_mqa_decode_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/flash_mqa_decode/CMakeLists.txt
# (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_FLASH_MQA_DECODE_NANOBIND_SRCS flash_mqa_decode_nanobind.cpp)
