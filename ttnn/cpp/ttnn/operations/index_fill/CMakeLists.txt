add_library(ttnn_op_index_fill ${LIB_TYPE})
add_library(TTNN::Ops::IndexFill ALIAS ttnn_op_index_fill)

target_precompile_headers(ttnn_op_index_fill REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_index_fill)

set_target_properties(
    ttnn_op_index_fill
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_index_fill
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
        device/index_fill_device_operation.cpp
        device/index_fill_multi_core_factory.cpp
        index_fill.cpp
)

target_include_directories(ttnn_op_index_fill PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_index_fill
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_index_fill
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/index_fill
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_index_fill LIBRARY COMPONENT tar)
