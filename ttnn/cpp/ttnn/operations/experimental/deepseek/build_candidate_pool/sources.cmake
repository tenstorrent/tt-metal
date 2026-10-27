# Source files for ttnn_op_experimental_deepseek_build_candidate_pool.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_BUILD_CANDIDATE_POOL_API_HEADERS build_candidate_pool.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_BUILD_CANDIDATE_POOL_SRCS
    build_candidate_pool.cpp
    device/build_candidate_pool_device_operation.cpp
    device/build_candidate_pool_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/build_candidate_pool/CMakeLists.txt
# (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_BUILD_CANDIDATE_POOL_NANOBIND_SRCS build_candidate_pool_nanobind.cpp)
