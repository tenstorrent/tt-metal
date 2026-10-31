set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ATTN_RES_BLOCK_MIX_API_HEADERS attn_res_block_mix.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ATTN_RES_BLOCK_MIX_SRCS
    device/attn_res_block_mix_device_operation.cpp
    device/attn_res_block_mix_program_factory.cpp
    attn_res_block_mix.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ATTN_RES_BLOCK_MIX_NANOBIND_SRCS attn_res_block_mix_nanobind.cpp)
