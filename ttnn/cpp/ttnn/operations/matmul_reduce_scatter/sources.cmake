# Source files for ttnn_op_matmul_reduce_scatter.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_MATMUL_REDUCE_SCATTER_API_HEADERS matmul_reduce_scatter.hpp)

set(TTNN_OP_MATMUL_REDUCE_SCATTER_SRCS
    matmul_reduce_scatter.cpp
    device/matmul_reduce_scatter_device_operation.cpp
    device/matmul_reduce_scatter_program_factory.cpp
    device/matmul_reduce_scatter_planner.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/matmul_reduce_scatter/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_MATMUL_REDUCE_SCATTER_NANOBIND_SRCS matmul_reduce_scatter_nanobind.cpp)
