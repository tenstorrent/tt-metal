include(sources.cmake)

add_library(ttnn_op_experimental_indexer_score ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::IndexerScore ALIAS ttnn_op_experimental_indexer_score)

tt_reuse_precompile_headers(ttnn_op_experimental_indexer_score TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_indexer_score)

set_target_properties(
    ttnn_op_experimental_indexer_score
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_indexer_score
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_INDEXER_SCORE_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_INDEXER_SCORE_SRCS}
)

target_link_libraries(ttnn_op_experimental_indexer_score PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_indexer_score FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
