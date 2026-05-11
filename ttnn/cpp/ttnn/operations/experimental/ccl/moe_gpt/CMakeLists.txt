include(sources.cmake)

add_library(ttnn_op_experimental_moe_gpt ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MoEGPT ALIAS ttnn_op_experimental_moe_gpt)

tt_reuse_precompile_headers(ttnn_op_experimental_moe_gpt TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_moe_gpt)

target_sources(ttnn_op_experimental_moe_gpt PRIVATE ${TTNN_OP_EXPERIMENTAL_MOE_GPT_SRCS})

target_include_directories(ttnn_op_experimental_moe_gpt PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_moe_gpt
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_moe_gpt LIBRARY COMPONENT tar)
