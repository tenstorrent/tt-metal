add_library(ttnn_op_experimental_reshape ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Reshape ALIAS ttnn_op_experimental_reshape)

target_precompile_headers(ttnn_op_experimental_reshape REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_reshape)

target_sources(ttnn_op_experimental_reshape PRIVATE view.cpp)

target_include_directories(ttnn_op_experimental_reshape PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_reshape
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_reshape LIBRARY COMPONENT tar)
