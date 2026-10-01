# Source files for ttnn_op_experimental_deepseek_width_sharded_all_reduce.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_WIDTH_SHARDED_ALL_REDUCE_API_HEADERS width_sharded_all_reduce.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_WIDTH_SHARDED_ALL_REDUCE_SRCS
    width_sharded_all_reduce.cpp
    device/width_sharded_all_reduce_device_operation.cpp
    device/width_sharded_all_reduce_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/width_sharded_all_reduce/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_WIDTH_SHARDED_ALL_REDUCE_NANOBIND_SRCS width_sharded_all_reduce_nanobind.cpp)
