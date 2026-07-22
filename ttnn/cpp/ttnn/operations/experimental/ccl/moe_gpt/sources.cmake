# Source files for ttnn_op_experimental_moe_gpt.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_MOE_GPT_API_HEADERS moe_gpt.hpp)

set(TTNN_OP_EXPERIMENTAL_MOE_GPT_SRCS
    moe_gpt.cpp
    device/moe_gpt_device_operation.cpp
    device/moe_gpt_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_MOE_GPT_NANOBIND_SRCS moe_gpt_nanobind.cpp)
