# Source files for ttnn_op_experimental_deepseek_prefill_moe_padding_config.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_PADDING_CONFIG_API_HEADERS moe_padding_config.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_PADDING_CONFIG_SRCS
    device/moe_padding_config_device_operation.cpp
    moe_padding_config.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_PADDING_CONFIG_NANOBIND_SRCS moe_padding_config_nanobind.cpp)
