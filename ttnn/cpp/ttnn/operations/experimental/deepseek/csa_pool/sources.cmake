# Source files for ttnn_op_experimental_deepseek_csa_pool.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_CSA_POOL_API_HEADERS csa_pool.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_CSA_POOL_SRCS
    csa_pool.cpp
    device/csa_pool_device_operation.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/csa_pool/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_CSA_POOL_NANOBIND_SRCS csa_pool_nanobind.cpp)
