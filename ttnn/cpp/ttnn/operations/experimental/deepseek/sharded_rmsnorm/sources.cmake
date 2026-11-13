# Source files for ttnn_op_experimental_deepseek_sharded_rmsnorm.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_SHARDED_RMSNORM_API_HEADERS sharded_rmsnorm.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_SHARDED_RMSNORM_SRCS
    sharded_rmsnorm.cpp
    device/sharded_rmsnorm_device_operation.cpp
    device/sharded_rmsnorm_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/sharded_rmsnorm/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_SHARDED_RMSNORM_NANOBIND_SRCS sharded_rmsnorm_nanobind.cpp)
