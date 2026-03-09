add_library(ttnn_op_experimental_deepseek_mla_mla_wqkv_ab ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Deepseek::MLA::MlaWqkvAb ALIAS ttnn_op_experimental_deepseek_mla_mla_wqkv_ab)

target_precompile_headers(ttnn_op_experimental_deepseek_mla_mla_wqkv_ab REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_mla_mla_wqkv_ab)

target_sources(
    ttnn_op_experimental_deepseek_mla_mla_wqkv_ab
    PRIVATE
        device/mla_wqkv_ab_device_operation.cpp
        device/mla_wqkv_ab_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_mla_mla_wqkv_ab PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_mla_mla_wqkv_ab
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_mla_mla_wqkv_ab LIBRARY COMPONENT tar)
