add_library(ttnn_op_embedding_backward ${LIB_TYPE})
add_library(TTNN::Ops::Embedding::Backward ALIAS ttnn_op_embedding_backward)

target_precompile_headers(ttnn_op_embedding_backward REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_embedding_backward)

set_target_properties(
    ttnn_op_embedding_backward
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_embedding_backward
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES embedding_backward.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        embedding_backward.cpp
        device/embedding_backward_device_operation.cpp
        device/embedding_backward_program_factory.cpp
)

target_include_directories(ttnn_op_embedding_backward PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_embedding_backward PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_embedding_backward
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/embedding_backward
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_embedding_backward LIBRARY COMPONENT tar)
