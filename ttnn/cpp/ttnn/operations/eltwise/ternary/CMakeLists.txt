add_library(ttnn_op_eltwise_ternary ${LIB_TYPE})
add_library(TTNN::Ops::Eltwise::Ternary ALIAS ttnn_op_eltwise_ternary)

target_precompile_headers(ttnn_op_eltwise_ternary REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_eltwise_ternary)
set_target_properties(
    ttnn_op_eltwise_ternary
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels where/device/kernels/*)
target_sources(
    ttnn_op_eltwise_ternary
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES where/where.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ternary_composite_op.cpp
        where/where.cpp
        where/device/where_device_operation.cpp
        where/device/where_program_factory.cpp
        where/device/where_utils.cpp
)

target_include_directories(ttnn_op_eltwise_ternary PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_eltwise_ternary
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_eltwise_ternary
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/eltwise/ternary
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_eltwise_ternary LIBRARY COMPONENT tar)
