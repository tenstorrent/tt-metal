add_library(ttnn_op_experimental_adaptive_pool ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::AdaptivePool ALIAS ttnn_op_experimental_adaptive_pool)

target_precompile_headers(ttnn_op_experimental_adaptive_pool REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_adaptive_pool)

target_sources(
    ttnn_op_experimental_adaptive_pool
    PRIVATE
        adaptive_pools.cpp
        adaptive_pool_utils.cpp
)

target_include_directories(ttnn_op_experimental_adaptive_pool PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_adaptive_pool
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_adaptive_pool LIBRARY COMPONENT tar)
