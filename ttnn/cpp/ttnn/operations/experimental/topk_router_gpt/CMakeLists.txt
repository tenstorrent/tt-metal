# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

add_library(ttnn_op_experimental_topk_router_gpt ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::TopkRouterGpt ALIAS ttnn_op_experimental_topk_router_gpt)

target_precompile_headers(ttnn_op_experimental_topk_router_gpt REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_topk_router_gpt)

target_sources(
    ttnn_op_experimental_topk_router_gpt
    PRIVATE
        device/topk_router_gpt_device_operation.cpp
        device/topk_router_gpt_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_topk_router_gpt PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_topk_router_gpt
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_topk_router_gpt LIBRARY COMPONENT tar)
