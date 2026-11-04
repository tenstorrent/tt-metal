# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::NomicEmbedTextV2Moe::GroupedMatmul
    ALIAS ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul
)

tt_reuse_precompile_headers(ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul)

set_target_properties(
    ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/grouped_matmul
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

# The Python binding source joins the shared `ttnn` module target, which exists only when
# WITH_PYTHON_BINDINGS is on.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_NANOBIND_SRCS})
endif()
