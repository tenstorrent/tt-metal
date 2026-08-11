include(sources.cmake)

add_library(ttnn_op_experimental_kda_qkv_causal_conv1d_silu ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::KDA::QkvCausalConv1dSilu ALIAS ttnn_op_experimental_kda_qkv_causal_conv1d_silu)

tt_reuse_precompile_headers(ttnn_op_experimental_kda_qkv_causal_conv1d_silu TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_kda_qkv_causal_conv1d_silu)

set_target_properties(
    ttnn_op_experimental_kda_qkv_causal_conv1d_silu
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_kda_qkv_causal_conv1d_silu
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_KDA_QKV_CAUSAL_CONV1D_SILU_API_HEADERS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_KDA_QKV_CAUSAL_CONV1D_SILU_SRCS}
)

target_link_libraries(ttnn_op_experimental_kda_qkv_causal_conv1d_silu PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_kda_qkv_causal_conv1d_silu FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_KDA_QKV_CAUSAL_CONV1D_SILU_NANOBIND_SRCS})
endif()
