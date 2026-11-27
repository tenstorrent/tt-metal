# Source files for ttnn_op_grouped_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_GROUPED_MATMUL_API_HEADERS grouped_matmul.hpp)

set(TTNN_OP_GROUPED_MATMUL_SRCS
    grouped_matmul.cpp
    device/grouped_matmul_device_operation.cpp
    device/grouped_matmul_program_factory.cpp
    device/grouped_matmul_planner.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/grouped_matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_GROUPED_MATMUL_NANOBIND_SRCS grouped_matmul_nanobind.cpp)
