add_library(ttnn_op_eltwise_quantization ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Quantization ALIAS ttnn_op_eltwise_quantization)

target_precompile_headers(ttnn_op_eltwise_quantization REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_quantization)

target_sources(ttnn_op_eltwise_quantization PRIVATE quantization.cpp)

target_include_directories(ttnn_op_eltwise_quantization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_quantization
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_eltwise_quantization LIBRARY COMPONENT tar)
