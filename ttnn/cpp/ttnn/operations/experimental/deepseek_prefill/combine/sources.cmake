# Source files for ttnn_op_experimental_deepseek_prefill_combine.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_API_HEADERS combine.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_SRCS
    device/combine_device_operation.cpp
    device/combine_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMBINE_NANOBIND_SRCS combine_nanobind.cpp)
