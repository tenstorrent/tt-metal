# Source files for ttnn_op_experimental_deepseek_prefill_masked_per_token_cast_back.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MASKED_PER_TOKEN_CAST_BACK_API_HEADERS masked_per_token_cast_back.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MASKED_PER_TOKEN_CAST_BACK_SRCS
    device/masked_per_token_cast_back_device_operation.cpp
    device/masked_per_token_cast_back_program_factory.cpp
    masked_per_token_cast_back.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_per_token_cast_back/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MASKED_PER_TOKEN_CAST_BACK_NANOBIND_SRCS
    masked_per_token_cast_back_nanobind.cpp
)
