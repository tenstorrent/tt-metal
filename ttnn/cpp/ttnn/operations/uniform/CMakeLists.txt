add_library(ttnn_op_uniform ${LIB_TYPE})
add_library(TTNN::Ops::Uniform ALIAS ttnn_op_uniform)

target_precompile_headers(ttnn_op_uniform REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_uniform)

# Only kernels FILE_SET; disable interface header verification
set_target_properties(
    ttnn_op_uniform
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_uniform
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
        device/uniform_device_operation.cpp
        device/uniform_program_factory.cpp
        uniform.cpp
)

target_include_directories(ttnn_op_uniform PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_uniform
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_uniform
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/uniform
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_uniform LIBRARY COMPONENT tar)
