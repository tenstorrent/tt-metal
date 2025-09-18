add_library(ttnn_op_embedding ${LIB_TYPE})
add_library(TTNN::Ops::Embedding ALIAS ttnn_op_embedding)

target_precompile_headers(ttnn_op_embedding REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_embedding)

set_target_properties(
    ttnn_op_embedding
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_embedding
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
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

install(
    TARGETS
        ttnn_op_embedding
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/embedding
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_embedding LIBRARY COMPONENT tar)
