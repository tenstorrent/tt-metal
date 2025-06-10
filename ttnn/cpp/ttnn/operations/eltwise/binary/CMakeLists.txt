add_library(ttnn_op_eltwise_binary ${LIB_TYPE})
add_library(TT::NN::Ops::Eltwise::Binary ALIAS ttnn_op_eltwise_binary)

target_precompile_headers(ttnn_op_eltwise_binary REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_binary)

target_sources(
    ttnn_op_eltwise_binary
    PRIVATE
        binary.cpp
        common/binary_op_utils.cpp
        device/binary_composite_op.cpp
        device/binary_device_operation.cpp
        device/broadcast_height_and_width_multi_core_program_factory.cpp
        device/broadcast_height_multi_core_program_factory.cpp
        device/broadcast_width_multi_core_program_factory.cpp
        device/element_wise_multi_core_program_factory.cpp
        device/element_wise_multi_core_sfpu_pgm_factory.cpp
        device/broadcast_height_multi_core_sharded_optimized_program_factory.cpp
        device/broadcast_height_multi_core_sharded_program_factory.cpp
)

target_include_directories(ttnn_op_eltwise_binary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_binary
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_eltwise_binary LIBRARY COMPONENT tar)
