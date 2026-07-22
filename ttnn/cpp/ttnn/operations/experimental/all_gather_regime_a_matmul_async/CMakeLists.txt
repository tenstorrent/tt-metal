include(sources.cmake)

add_library(ttnn_op_experimental_all_gather_regime_a_matmul_async ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::AllGatherRegimeAMatmulAsync ALIAS ttnn_op_experimental_all_gather_regime_a_matmul_async)

tt_reuse_precompile_headers(ttnn_op_experimental_all_gather_regime_a_matmul_async TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_all_gather_regime_a_matmul_async)

set_target_properties(
    ttnn_op_experimental_all_gather_regime_a_matmul_async
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_all_gather_regime_a_matmul_async
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_all_gather_regime_a_matmul_async
    PRIVATE TT::Metalium
    PUBLIC
        TTNN::Core
        TTNN::Ops::Experimental::RegimeAMatmul
)

install(TARGETS ttnn_op_experimental_all_gather_regime_a_matmul_async FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn` Python module target.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_NANOBIND_SRCS})
endif()
