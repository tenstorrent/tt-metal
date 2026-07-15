# Source files for ttnn_op_experimental_deepseek_prefill_post_combine_reduce.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_POST_COMBINE_REDUCE_API_HEADERS post_combine_reduce.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_POST_COMBINE_REDUCE_SRCS
    device/post_combine_reduce_device_operation.cpp
    device/post_combine_reduce_program_factory.cpp
    post_combine_reduce.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_POST_COMBINE_REDUCE_NANOBIND_SRCS post_combine_reduce_nanobind.cpp)
