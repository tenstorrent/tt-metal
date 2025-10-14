add_library(ttnn_op_eltwise_fused ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Fused ALIAS ttnn_op_eltwise_fused)

target_precompile_headers(ttnn_op_eltwise_fused REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_fused)
set_target_properties(
    ttnn_op_eltwise_fused
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            false
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_eltwise_fused
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES device/materialize_device_operation.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/materialize_device_operation.cpp
        device/materialize_program_factory.cpp
)

target_include_directories(ttnn_op_eltwise_fused PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_fused
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_eltwise_fused
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/fused
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_eltwise_fused LIBRARY COMPONENT tar)
