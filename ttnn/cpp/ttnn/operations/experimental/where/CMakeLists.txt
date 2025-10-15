add_library(ttnn_op_experimental_where ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Where ALIAS ttnn_op_experimental_where)

target_precompile_headers(ttnn_op_experimental_where REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_where)

set_target_properties(
    ttnn_op_experimental_where
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_where
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
        device/where_device_operation.cpp
        device/program_factory/element_wise_multi_core_where_program.cpp
)

target_include_directories(ttnn_op_experimental_where PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_where
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_where
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/where
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_where LIBRARY COMPONENT tar)
