# Source files for ttnn_op_pair_pool.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_PAIR_POOL_API_HEADERS pair_pool.hpp)

set(TTNN_OP_PAIR_POOL_SRCS
    pair_pool.cpp
    device/pair_pool_device_operation.cpp
    device/pair_pool_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/pair_pool/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_PAIR_POOL_NANOBIND_SRCS pair_pool_nanobind.cpp)
