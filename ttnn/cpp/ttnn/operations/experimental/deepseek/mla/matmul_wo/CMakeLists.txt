add_library(ttnn_op_experimental_deepseek_mla_matmul_wo ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Deepseek::MLA::Matmul_WO ALIAS ttnn_op_experimental_deepseek_mla_matmul_wo)

target_precompile_headers(ttnn_op_experimental_deepseek_mla_matmul_wo REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_mla_matmul_wo)

target_sources(
    ttnn_op_experimental_deepseek_mla_matmul_wo
    PRIVATE
        matmul_wo.cpp
        device/matmul_wo_device_operation.cpp
        device/matmul_wo_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_mla_matmul_wo PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_mla_matmul_wo
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_mla_matmul_wo LIBRARY COMPONENT tar)
