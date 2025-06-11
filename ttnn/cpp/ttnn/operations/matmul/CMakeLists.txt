add_library(ttnn_op_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Matmul ALIAS ttnn_op_matmul)

target_precompile_headers(ttnn_op_matmul REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_matmul)

target_sources(
    ttnn_op_matmul
    PRIVATE
        device/matmul_op.cpp
        device/matmul_op_multi_core_program_factory.cpp
        device/matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp
        device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp
        device/matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp
        device/matmul_op_multi_core_reuse_optimized_program_factory.cpp
        device/matmul_op_multi_core_reuse_program_factory.cpp
        matmul.cpp
)

target_include_directories(ttnn_op_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_matmul
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_matmul LIBRARY COMPONENT tar)
