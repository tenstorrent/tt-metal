add_library(ttnn_op_experimental_where ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Where ALIAS ttnn_op_experimental_where)

target_precompile_headers(ttnn_op_experimental_where REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_where)

target_sources(
    ttnn_op_experimental_where
    PRIVATE
        device/where_device_operation.cpp
        device/program_factory/element_wise_multi_core_where_program.cpp
)

target_include_directories(ttnn_op_experimental_where PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_where
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_where LIBRARY COMPONENT tar)
