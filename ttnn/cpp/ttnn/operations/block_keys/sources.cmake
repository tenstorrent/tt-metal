# Source files for ttnn_op_block_keys.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BLOCK_KEYS_API_HEADERS block_keys.hpp)

set(TTNN_OP_BLOCK_KEYS_SRCS
    block_keys.cpp
    device/block_keys_device_operation.cpp
    device/block_keys_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/block_keys/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_BLOCK_KEYS_NANOBIND_SRCS block_keys_nanobind.cpp)
