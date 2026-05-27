include(sources.cmake)

add_library(ttnn_op_experimental_moe_gpt ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MoEGPT ALIAS ttnn_op_experimental_moe_gpt)

tt_reuse_precompile_headers(ttnn_op_experimental_moe_gpt TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_moe_gpt)

set_target_properties(
    ttnn_op_experimental_moe_gpt
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_moe_gpt
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_MOE_GPT_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_MOE_GPT_SRCS}
)

target_link_libraries(ttnn_op_experimental_moe_gpt PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_moe_gpt FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
