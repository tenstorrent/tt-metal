include(sources.cmake)

add_library(ttnn_op_experimental_adaptive_pool ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::AdaptivePool ALIAS ttnn_op_experimental_adaptive_pool)

tt_reuse_precompile_headers(ttnn_op_experimental_adaptive_pool TTNN::PCHLean)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_adaptive_pool)

set_target_properties(
    ttnn_op_experimental_adaptive_pool
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_adaptive_pool
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_ADAPTIVE_POOL_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_ADAPTIVE_POOL_SRCS}
)

target_link_libraries(ttnn_op_experimental_adaptive_pool PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_adaptive_pool FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
