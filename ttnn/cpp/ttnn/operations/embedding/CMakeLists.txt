add_library(ttnn_op_embedding ${LIB_TYPE})
add_library(TTNN::Ops::Embedding ALIAS ttnn_op_embedding)

target_precompile_headers(ttnn_op_embedding REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_embedding)

target_sources(
    ttnn_op_embedding
    PRIVATE
        device/embedding_device_operation.cpp
        embedding.cpp
)

target_include_directories(ttnn_op_embedding PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_embedding
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_embedding LIBRARY COMPONENT tar)
