add_library(ttnn_op_eltwise_complex_unary ${LIB_TYPE})
add_library(TT::NN::Ops::Eltwise::Complex::Unary ALIAS ttnn_op_eltwise_complex_unary)

target_precompile_headers(ttnn_op_eltwise_complex_unary REUSE_FROM TT::CommonPCH)

target_sources(ttnn_op_eltwise_complex_unary PRIVATE device/complex_unary_op.cpp)

target_include_directories(ttnn_op_eltwise_complex_unary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_complex_unary
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_eltwise_complex_unary LIBRARY COMPONENT tar)
