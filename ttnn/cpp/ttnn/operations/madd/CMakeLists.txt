add_library(ttnn_op_madd ${LIB_TYPE})
add_library(TTNN::Ops::MAdd ALIAS ttnn_op_madd)

target_precompile_headers(ttnn_op_madd REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_madd)

set_target_properties(
    ttnn_op_madd
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_madd
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/madd_device_operation.cpp
        device/madd_program_factory.cpp
        device/madd_program_factory_sharded.cpp
        madd.cpp
)

target_include_directories(ttnn_op_madd PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_madd
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_madd
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/madd
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_madd LIBRARY COMPONENT tar)
