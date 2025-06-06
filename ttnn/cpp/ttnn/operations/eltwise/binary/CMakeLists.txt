add_library(ttnn_op_eltwise_binary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Binary ALIAS ttnn_op_eltwise_binary)

target_precompile_headers(ttnn_op_eltwise_binary REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_binary)
set_target_properties(
    ttnn_op_eltwise_binary
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

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

file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_eltwise_binary
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
)

target_include_directories(ttnn_op_eltwise_binary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_binary
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_eltwise_binary
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/binary
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_binary LIBRARY COMPONENT tar)
