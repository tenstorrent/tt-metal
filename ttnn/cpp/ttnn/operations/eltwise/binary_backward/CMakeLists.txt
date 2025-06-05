add_library(ttnn_op_eltwise_binary_backward ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Binary::Backward ALIAS ttnn_op_eltwise_binary_backward)

target_precompile_headers(ttnn_op_eltwise_binary_backward REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_binary_backward)

target_sources(ttnn_op_eltwise_binary_backward PRIVATE binary_backward.cpp)

target_include_directories(ttnn_op_eltwise_binary_backward PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_binary_backward
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_eltwise_binary_backward LIBRARY COMPONENT tar)
