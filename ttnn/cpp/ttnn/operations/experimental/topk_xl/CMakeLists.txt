# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(ttnn_op_experimental_topk_xl ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::TopkXL ALIAS ttnn_op_experimental_topk_xl)

target_precompile_headers(ttnn_op_experimental_topk_xl REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_topk_xl)

set_target_properties(
    ttnn_op_experimental_topk_xl
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_topk_xl
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_TOPK_XL_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_TOPK_XL_SRCS}
)

target_link_libraries(ttnn_op_experimental_topk_xl PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_topk_xl FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
