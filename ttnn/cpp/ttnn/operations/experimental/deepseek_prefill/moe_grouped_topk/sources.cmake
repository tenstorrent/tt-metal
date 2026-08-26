set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_GROUPED_TOPK_API_HEADERS moe_grouped_topk.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_GROUPED_TOPK_SRCS
    device/moe_grouped_topk_device_operation.cpp
    device/moe_grouped_topk_program_factory.cpp
    moe_grouped_topk.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_GROUPED_TOPK_NANOBIND_SRCS moe_grouped_topk_nanobind.cpp)
