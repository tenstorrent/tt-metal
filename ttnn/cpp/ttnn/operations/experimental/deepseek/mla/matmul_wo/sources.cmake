# Source files for ttnn_op_experimental_deepseek_mla_matmul_wo.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MLA_MATMUL_WO_API_HEADERS matmul_wo.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MLA_MATMUL_WO_SRCS
    matmul_wo.cpp
    device/matmul_wo_device_operation.cpp
    device/matmul_wo_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MLA_MATMUL_WO_NANOBIND_SRCS matmul_wo_nanobind.cpp)
