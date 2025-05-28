add_library(ttnn_op_eltwise_quantization ${LIB_TYPE})
add_library(TT::NN::Ops::Eltwise::Quantization ALIAS ttnn_op_eltwise_quantization)

target_precompile_headers(ttnn_op_eltwise_quantization REUSE_FROM TT::CommonPCH)

target_sources(ttnn_op_eltwise_quantization PRIVATE quantization.cpp)

target_include_directories(ttnn_op_eltwise_quantization PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_quantization
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_eltwise_quantization LIBRARY COMPONENT tar)
