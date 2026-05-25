include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_iterative_topk ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::IterativeTopk
    ALIAS ttnn_op_experimental_deepseek_prefill_iterative_topk
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_iterative_topk REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_iterative_topk)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_iterative_topk
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_iterative_topk
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ITERATIVE_TOPK_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ITERATIVE_TOPK_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_iterative_topk PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_iterative_topk
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/iterative_topk
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_iterative_topk
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)
