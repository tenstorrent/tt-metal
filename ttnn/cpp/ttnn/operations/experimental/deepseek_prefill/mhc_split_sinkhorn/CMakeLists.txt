include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::MhcSplitSinkhorn
    ALIAS ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_SPLIT_SINKHORN_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_SPLIT_SINKHORN_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)
