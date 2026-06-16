# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(ttnn_op_experimental_topk_large_indices ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::TopkLargeIndices ALIAS ttnn_op_experimental_topk_large_indices)

target_precompile_headers(ttnn_op_experimental_topk_large_indices REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_topk_large_indices)

set_target_properties(
    ttnn_op_experimental_topk_large_indices
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_topk_large_indices
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_TOPK_LARGE_INDICES_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_TOPK_LARGE_INDICES_SRCS}
)

target_link_libraries(ttnn_op_experimental_topk_large_indices PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_topk_large_indices FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
