include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_high_bw_all_gather ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::HighBwAllGather
    ALIAS ttnn_op_experimental_deepseek_prefill_high_bw_all_gather
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_high_bw_all_gather REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_high_bw_all_gather)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_high_bw_all_gather
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)
target_sources(
    ttnn_op_experimental_deepseek_prefill_high_bw_all_gather
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HIGH_BW_ALL_GATHER_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HIGH_BW_ALL_GATHER_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_high_bw_all_gather PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_high_bw_all_gather
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/high_bw_all_gather
        COMPONENT ttnn-runtime
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_HIGH_BW_ALL_GATHER_NANOBIND_SRCS})
endif()
