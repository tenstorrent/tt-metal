include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_mla_matmul_wo ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Deepseek::MLA::Matmul_WO ALIAS ttnn_op_experimental_deepseek_mla_matmul_wo)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_mla_matmul_wo TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_mla_matmul_wo)

target_sources(ttnn_op_experimental_deepseek_mla_matmul_wo PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MLA_MATMUL_WO_SRCS})

target_include_directories(ttnn_op_experimental_deepseek_mla_matmul_wo PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_mla_matmul_wo
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_mla_matmul_wo LIBRARY COMPONENT tar)
