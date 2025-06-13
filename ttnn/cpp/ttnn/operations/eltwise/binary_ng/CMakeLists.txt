add_library(ttnn_op_eltwise_binary_ng ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Binary::NG ALIAS ttnn_op_eltwise_binary_ng)

target_precompile_headers(ttnn_op_eltwise_binary_ng REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_binary_ng)
set_target_properties(
    ttnn_op_eltwise_binary_ng
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

target_sources(
    ttnn_op_eltwise_binary_ng
    PRIVATE
        device/binary_ng_device_operation.cpp
        device/binary_ng_program_factory.cpp
        device/binary_ng_utils.cpp
)
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_eltwise_binary_ng
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
)

target_include_directories(ttnn_op_eltwise_binary_ng PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_binary_ng
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_eltwise_binary_ng
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/binary_ng
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_eltwise_binary_ng LIBRARY COMPONENT tar)
