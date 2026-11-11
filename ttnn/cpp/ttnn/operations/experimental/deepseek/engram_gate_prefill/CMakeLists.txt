include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_engram_gate_prefill ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeek::EngramGatePrefill
    ALIAS ttnn_op_experimental_deepseek_engram_gate_prefill
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_engram_gate_prefill TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_engram_gate_prefill)

set_target_properties(
    ttnn_op_experimental_deepseek_engram_gate_prefill
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_engram_gate_prefill
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_PREFILL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_PREFILL_SRCS}
)

target_link_libraries(ttnn_op_experimental_deepseek_engram_gate_prefill PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_deepseek_engram_gate_prefill
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate_prefill
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_engram_gate_prefill FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_ENGRAM_GATE_PREFILL_NANOBIND_SRCS})
endif()
