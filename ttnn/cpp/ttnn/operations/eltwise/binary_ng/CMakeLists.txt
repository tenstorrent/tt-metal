add_library(ttnn_op_eltwise_binary_ng ${LIB_TYPE})
add_library(TT::NN::Ops::Eltwise::Binary::NG ALIAS ttnn_op_eltwise_binary_ng)

target_precompile_headers(ttnn_op_eltwise_binary_ng REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_eltwise_binary_ng
    PRIVATE
        device/binary_ng_device_operation.cpp
        device/binary_ng_program_factory.cpp
        device/binary_ng_utils.cpp
)

target_include_directories(ttnn_op_eltwise_binary_ng PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_binary_ng
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_eltwise_binary_ng LIBRARY COMPONENT tar)
