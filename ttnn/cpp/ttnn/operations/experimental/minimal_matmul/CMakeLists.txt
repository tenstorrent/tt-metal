add_library(ttnn_op_experimental_minimal_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MinimalMatmul ALIAS ttnn_op_experimental_minimal_matmul)

target_precompile_headers(ttnn_op_experimental_minimal_matmul REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_minimal_matmul)

set_target_properties(
    ttnn_op_experimental_minimal_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_minimal_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        minimal_matmul.cpp
        minimal_matmul_split.cpp
        device/minimal_matmul_device_operation.cpp
        device/minimal_matmul_program_factory.cpp
)

# Disable unity build for program factory files to avoid symbol collisions
set_source_files_properties(
    device/minimal_matmul_program_factory.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

target_include_directories(ttnn_op_experimental_minimal_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_minimal_matmul
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_minimal_matmul
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/minimal_matmul
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_minimal_matmul LIBRARY COMPONENT tar)
