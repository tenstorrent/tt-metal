# Source files for ttnn_op_all_to_all_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ALL_TO_ALL_MATMUL_API_HEADERS all_to_all_matmul.hpp)

set(TTNN_OP_ALL_TO_ALL_MATMUL_SRCS
    all_to_all_matmul.cpp
    device/all_to_all_matmul_device_operation.cpp
    device/all_to_all_matmul_program_factory.cpp
    device/all_to_all_matmul_planner.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/all_to_all_matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_ALL_TO_ALL_MATMUL_NANOBIND_SRCS all_to_all_matmul_nanobind.cpp)
