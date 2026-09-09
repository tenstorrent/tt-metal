# Source files for ttnn_op_experimental_deepseek_all_gather_for_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ALL_GATHER_FOR_MATMUL_API_HEADERS all_gather_for_matmul.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ALL_GATHER_FOR_MATMUL_SRCS
    all_gather_for_matmul.cpp
    device/all_gather_for_matmul_device_operation.cpp
    device/all_gather_for_matmul_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/all_gather_for_matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_ALL_GATHER_FOR_MATMUL_NANOBIND_SRCS all_gather_for_matmul_nanobind.cpp)
