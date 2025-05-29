add_library(ttnn_op_eltwise_ternary ${LIB_TYPE})
add_library(TT::NN::Ops::Eltwise::Ternary ALIAS ttnn_op_eltwise_ternary)

target_precompile_headers(ttnn_op_eltwise_ternary REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_eltwise_ternary
    PRIVATE
        ternary_composite_op.cpp
        where.cpp
)

target_include_directories(ttnn_op_eltwise_ternary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_ternary
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_eltwise_ternary LIBRARY COMPONENT tar)
