set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ATTN_RES_WEIGHTED_REDUCE_NC_API_HEADERS attn_res_weighted_reduce_nc.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ATTN_RES_WEIGHTED_REDUCE_NC_SRCS
    device/attn_res_weighted_reduce_nc_device_operation.cpp
    device/attn_res_weighted_reduce_nc_program_factory.cpp
    attn_res_weighted_reduce_nc.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ATTN_RES_WEIGHTED_REDUCE_NC_NANOBIND_SRCS
    attn_res_weighted_reduce_nc_nanobind.cpp
)
