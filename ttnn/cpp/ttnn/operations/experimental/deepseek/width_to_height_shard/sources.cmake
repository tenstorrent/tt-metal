# Source files for ttnn_op_experimental_deepseek_width_to_height_shard.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_WIDTH_TO_HEIGHT_SHARD_API_HEADERS width_to_height_shard.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_WIDTH_TO_HEIGHT_SHARD_SRCS
    width_to_height_shard.cpp
    device/width_to_height_shard_device_operation.cpp
    device/width_to_height_shard_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/width_to_height_shard/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_WIDTH_TO_HEIGHT_SHARD_NANOBIND_SRCS width_to_height_shard_nanobind.cpp)
