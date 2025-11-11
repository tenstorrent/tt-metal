add_library(ttnn_op_eltwise_materialize ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Materialize ALIAS ttnn_op_eltwise_materialize)

target_precompile_headers(ttnn_op_eltwise_materialize REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_materialize)
set_target_properties(
    ttnn_op_eltwise_materialize
    PROPERTIES
        INTERFACE_HEADER_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_eltwise_materialize
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

target_include_directories(ttnn_op_eltwise_materialize PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_materialize
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_eltwise_materialize
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/materialize
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_eltwise_materialize LIBRARY COMPONENT tar)
