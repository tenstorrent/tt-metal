include(sources.cmake)

add_library(ttnn_op_embedding ${LIB_TYPE})
add_library(TTNN::Ops::Embedding ALIAS ttnn_op_embedding)

tt_reuse_precompile_headers(ttnn_op_embedding TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_embedding)

set_target_properties(
    ttnn_op_embedding
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_embedding
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EMBEDDING_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EMBEDDING_SRCS}
)

target_include_directories(ttnn_op_embedding PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_embedding PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_embedding
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/embedding
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_embedding LIBRARY COMPONENT tar)
