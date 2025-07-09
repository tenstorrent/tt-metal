add_library(ttnn_op_eltwise_unary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Unary ALIAS ttnn_op_eltwise_unary)

target_precompile_headers(ttnn_op_eltwise_unary REUSE_FROM TT::CommonPCH)
set_target_properties(
    ttnn_op_eltwise_unary
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

target_sources(
    ttnn_op_eltwise_unary
    PRIVATE
        common/unary_op_utils.cpp
        device/unary_composite_op.cpp
        device/unary_device_operation.cpp
        device/unary_program_factory.cpp
        device/unary_sharded_program_factory.cpp
        tanh_accurate/device/tanh_accurate_device_operation.cpp
        tanh_accurate/device/tanh_accurate_pgm_factory.cpp
        tanh_accurate/device/tanh_accurate_sharded_pgm_factory.cpp
        tanh_accurate/tanh_accurate.cpp
        unary.cpp
)

file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_eltwise_unary
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
)

target_include_directories(ttnn_op_eltwise_unary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_unary
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_eltwise_unary
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/unary
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_unary LIBRARY COMPONENT tar)
