add_library(ttnn_op_eltwise_complex_binary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Complex::Binary ALIAS ttnn_op_eltwise_complex_binary)

target_precompile_headers(ttnn_op_eltwise_complex_binary REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_complex_binary)

target_sources(ttnn_op_eltwise_complex_binary PRIVATE device/complex_binary_op.cpp)

target_include_directories(ttnn_op_eltwise_complex_binary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_complex_binary
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_eltwise_complex_binary LIBRARY COMPONENT tar)
