add_library(ttnn_op_experimental_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Matmul ALIAS ttnn_op_experimental_matmul)

target_precompile_headers(ttnn_op_experimental_matmul REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_matmul)

target_sources(
    ttnn_op_experimental_matmul
    PRIVATE
        attn_matmul/attn_matmul.cpp
        attn_matmul/device/attn_matmul_device_operation.cpp
        attn_matmul/device/attn_matmul_program_factory.cpp
        group_attn_matmul/device/group_attn_matmul_device_operation.cpp
        group_attn_matmul/device/group_attn_matmul_program_factory.cpp
        group_attn_matmul/group_attn_matmul.cpp
)

target_include_directories(ttnn_op_experimental_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_matmul
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_matmul LIBRARY COMPONENT tar)
