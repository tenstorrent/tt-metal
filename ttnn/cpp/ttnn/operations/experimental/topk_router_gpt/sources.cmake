# Source files for ttnn_op_experimental_topk_router_gpt.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_TOPK_ROUTER_GPT_API_HEADERS
    topk_router_gpt.hpp
    device/topk_router_gpt_device_operation.hpp
    device/topk_router_gpt_device_operation_types.hpp
    device/topk_router_gpt_program_factory.hpp
)

set(TTNN_OP_EXPERIMENTAL_TOPK_ROUTER_GPT_SRCS
    device/topk_router_gpt_device_operation.cpp
    device/topk_router_gpt_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/topk_router_gpt/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_TOPK_ROUTER_GPT_NANOBIND_SRCS topk_router_gpt_nanobind.cpp)
