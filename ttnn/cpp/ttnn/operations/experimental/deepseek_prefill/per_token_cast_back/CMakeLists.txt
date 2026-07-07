include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_per_token_cast_back ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::PerTokenCastBack
    ALIAS ttnn_op_experimental_deepseek_prefill_per_token_cast_back
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_per_token_cast_back REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_per_token_cast_back)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_per_token_cast_back
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_per_token_cast_back
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PER_TOKEN_CAST_BACK_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PER_TOKEN_CAST_BACK_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_per_token_cast_back PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_per_token_cast_back
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_per_token_cast_back
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)
