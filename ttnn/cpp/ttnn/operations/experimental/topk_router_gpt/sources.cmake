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
