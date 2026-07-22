# Source files for ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PER_TOKEN_CAST_TO_FP8_API_HEADERS per_token_cast_to_fp8.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PER_TOKEN_CAST_TO_FP8_SRCS
    device/per_token_cast_to_fp8_device_operation.cpp
    device/per_token_cast_to_fp8_program_factory.cpp
    per_token_cast_to_fp8.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PER_TOKEN_CAST_TO_FP8_NANOBIND_SRCS per_token_cast_to_fp8_nanobind.cpp)
