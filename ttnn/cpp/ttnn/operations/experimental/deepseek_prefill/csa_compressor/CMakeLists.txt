include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_csa_compressor ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::CsaCompressor
    ALIAS ttnn_op_experimental_deepseek_prefill_csa_compressor
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_csa_compressor REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_csa_compressor)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_csa_compressor
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)
target_sources(
    ttnn_op_experimental_deepseek_prefill_csa_compressor
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_CSA_COMPRESSOR_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_CSA_COMPRESSOR_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_csa_compressor
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
        TTNN::Ops::Experimental::DeepSeekPrefill::CompressorStateExchange
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_csa_compressor
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/csa_compressor
        COMPONENT ttnn-runtime
)
install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_csa_compressor
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_CSA_COMPRESSOR_NANOBIND_SRCS})
endif()
