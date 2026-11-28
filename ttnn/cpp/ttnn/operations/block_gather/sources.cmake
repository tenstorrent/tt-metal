# Source files for ttnn_op_block_gather.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BLOCK_GATHER_API_HEADERS block_gather.hpp)

set(TTNN_OP_BLOCK_GATHER_SRCS
    block_gather.cpp
    device/block_gather_device_operation.cpp
    device/block_gather_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/block_gather/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_BLOCK_GATHER_NANOBIND_SRCS block_gather_nanobind.cpp)
