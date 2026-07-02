include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_offset_cumsum ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::OffsetCumsum
    ALIAS ttnn_op_experimental_deepseek_prefill_offset_cumsum
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_offset_cumsum REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_offset_cumsum)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_offset_cumsum
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_offset_cumsum
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_OFFSET_CUMSUM_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_OFFSET_CUMSUM_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_offset_cumsum PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_offset_cumsum
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_offset_cumsum
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)
