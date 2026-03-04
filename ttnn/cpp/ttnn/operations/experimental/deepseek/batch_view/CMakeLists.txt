add_library(ttnn_op_experimental_deepseek_batch_view ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeek::BatchView ALIAS ttnn_op_experimental_deepseek_batch_view)

target_precompile_headers(ttnn_op_experimental_deepseek_batch_view REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_batch_view)

target_sources(ttnn_op_experimental_deepseek_batch_view PRIVATE batch_view.cpp)

target_include_directories(ttnn_op_experimental_deepseek_batch_view PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_batch_view
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_batch_view LIBRARY COMPONENT tar)
