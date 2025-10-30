add_library(ttnn_op_ax_plus_b ${LIB_TYPE})
add_library(TTNN::Ops::AxPlusB ALIAS ttnn_op_ax_plus_b)

target_precompile_headers(ttnn_op_ax_plus_b REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_ax_plus_b)
set_target_properties(
    ttnn_op_ax_plus_b
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_ax_plus_b
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES device/ax_plus_b_device_operation.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/ax_plus_b_device_operation.cpp
        device/ax_plus_b_program_factory_common.cpp
        device/single_core_program_factory.cpp
        device/multi_core_program_factory.cpp
)

target_include_directories(ttnn_op_ax_plus_b PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_ax_plus_b PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_ax_plus_b
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/ax_plus_b
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_ax_plus_b LIBRARY COMPONENT tar)
