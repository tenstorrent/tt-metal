add_library(ttnn_op_eltwise_unary_backward ${LIB_TYPE})
add_library(TT::NN::Ops::Eltwise::Unary::Backward ALIAS ttnn_op_eltwise_unary_backward)

target_precompile_headers(ttnn_op_eltwise_unary_backward REUSE_FROM TT::CommonPCH)

target_sources(ttnn_op_eltwise_unary_backward PRIVATE unary_backward.cpp)

target_include_directories(ttnn_op_eltwise_unary_backward PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_unary_backward
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_eltwise_unary_backward LIBRARY COMPONENT tar)
