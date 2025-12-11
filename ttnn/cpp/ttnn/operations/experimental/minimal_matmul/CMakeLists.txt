add_library(ttnn_op_experimental_minimal_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MinimalMatmul ALIAS ttnn_op_experimental_minimal_matmul)

target_precompile_headers(ttnn_op_experimental_minimal_matmul REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_minimal_matmul)

target_sources(
    ttnn_op_experimental_minimal_matmul
    PRIVATE
        minimal_matmul.cpp
        device/minimal_matmul_device_operation.cpp
        device/minimal_matmul_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_minimal_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_minimal_matmul
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_minimal_matmul LIBRARY COMPONENT tar)
