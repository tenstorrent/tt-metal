include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8 ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::PerTokenCastToFp8
    ALIAS ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8 REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PER_TOKEN_CAST_TO_FP8_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PER_TOKEN_CAST_TO_FP8_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_per_token_cast_to_fp8
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)
