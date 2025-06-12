add_library(ttnn_op_eltwise_ternary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Ternary ALIAS ttnn_op_eltwise_ternary)

target_precompile_headers(ttnn_op_eltwise_ternary REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_ternary)

target_sources(
    ttnn_op_eltwise_ternary
    PRIVATE
        ternary_composite_op.cpp
        where/where.cpp
        where/device/where_device_operation.cpp
        where/device/where_program_factory.cpp
)

target_include_directories(ttnn_op_eltwise_ternary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_ternary
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_eltwise_ternary LIBRARY COMPONENT tar)
