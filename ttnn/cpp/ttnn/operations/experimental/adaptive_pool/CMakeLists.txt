include(sources.cmake)

add_library(ttnn_op_experimental_adaptive_pool ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::AdaptivePool ALIAS ttnn_op_experimental_adaptive_pool)

tt_reuse_precompile_headers(ttnn_op_experimental_adaptive_pool TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_adaptive_pool)

target_sources(ttnn_op_experimental_adaptive_pool PRIVATE ${TTNN_OP_EXPERIMENTAL_ADAPTIVE_POOL_SRCS})

target_include_directories(ttnn_op_experimental_adaptive_pool PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_adaptive_pool
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_adaptive_pool LIBRARY COMPONENT tar)
