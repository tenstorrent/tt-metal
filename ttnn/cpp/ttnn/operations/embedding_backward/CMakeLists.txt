add_library(ttnn_op_embedding_backward ${LIB_TYPE})
add_library(TTNN::Ops::Embedding::Backward ALIAS ttnn_op_embedding_backward)

target_precompile_headers(ttnn_op_embedding_backward REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_embedding_backward)

target_sources(
    ttnn_op_embedding_backward
    PRIVATE
        embedding_backward.cpp
        device/embedding_backward_device_operation.cpp
        device/embedding_backward_program_factory.cpp
)

target_include_directories(ttnn_op_embedding_backward PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_embedding_backward
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_embedding_backward LIBRARY COMPONENT tar)
