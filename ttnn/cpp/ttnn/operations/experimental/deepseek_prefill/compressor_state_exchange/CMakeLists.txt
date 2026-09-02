include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_compressor_state_exchange ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::CompressorStateExchange
    ALIAS ttnn_op_experimental_deepseek_prefill_compressor_state_exchange
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_compressor_state_exchange REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_compressor_state_exchange)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_compressor_state_exchange
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_compressor_state_exchange
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMPRESSOR_STATE_EXCHANGE_API_HEADERS}
    PRIVATE
        compressor_state_exchange.cpp
        device/compressor_state_select.cpp
        device/compressor_state_select_program_factory.cpp
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_compressor_state_exchange
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
        TTNN::Ops::CCL
        TTNN::Ops::DataMovement
        TTNN::Ops::PointToPoint
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_compressor_state_exchange LIBRARY COMPONENT tar)
install(TARGETS ttnn_op_experimental_deepseek_prefill_compressor_state_exchange FILE_SET api COMPONENT ttnn-dev)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_COMPRESSOR_STATE_EXCHANGE_NANOBIND_SRCS})
endif()
