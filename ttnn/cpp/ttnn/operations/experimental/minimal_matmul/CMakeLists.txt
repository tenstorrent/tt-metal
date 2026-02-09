add_library(ttnn_op_experimental_minimal_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MinimalMatmul ALIAS ttnn_op_experimental_minimal_matmul)

target_precompile_headers(ttnn_op_experimental_minimal_matmul REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_minimal_matmul)

target_sources(
    ttnn_op_experimental_minimal_matmul
    PRIVATE
        minimal_matmul.cpp
        minimal_matmul_split.cpp
        device/minimal_matmul_device_operation.cpp
        device/minimal_matmul_program_factory.cpp
        device/minimal_matmul_split_device_operation.cpp
        device/minimal_matmul_split_program_factory.cpp
)

# Disable unity build for program factory files to avoid symbol collisions
set_source_files_properties(
    device/minimal_matmul_program_factory.cpp
    device/minimal_matmul_split_program_factory.cpp
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

install(TARGETS ttnn_op_experimental_minimal_matmul LIBRARY COMPONENT tar)
