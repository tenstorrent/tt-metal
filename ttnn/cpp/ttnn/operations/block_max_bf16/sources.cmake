# Source files for ttnn_op_block_max_bf16.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BLOCK_MAX_BF16_API_HEADERS block_max_bf16.hpp)

set(TTNN_OP_BLOCK_MAX_BF16_SRCS
    block_max_bf16.cpp
    device/block_max_bf16_device_operation.cpp
    device/block_max_bf16_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/block_max_bf16/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_BLOCK_MAX_BF16_NANOBIND_SRCS block_max_bf16_nanobind.cpp)
