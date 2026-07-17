# Source files for ttnn_op_experimental_deepseek_moe_post_combine_tilize.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_POST_COMBINE_TILIZE_API_HEADERS deepseek_moe_post_combine_tilize.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_POST_COMBINE_TILIZE_SRCS
    device/deepseek_moe_post_combine_tilize_device_operation.cpp
    device/deepseek_moe_post_combine_tilize_program_factory.cpp
    deepseek_moe_post_combine_tilize.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_tilize/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_POST_COMBINE_TILIZE_NANOBIND_SRCS deepseek_moe_post_combine_tilize_nanobind.cpp)
