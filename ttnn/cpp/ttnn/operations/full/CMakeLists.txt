add_library(ttnn_op_full ${LIB_TYPE})
add_library(TTNN::Ops::Full ALIAS ttnn_op_full)

target_precompile_headers(ttnn_op_full REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_full)

set_target_properties(
    ttnn_op_full
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_full
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
        device/full_device_operation.cpp
        device/full_program_factory.cpp
        full.cpp
)

target_include_directories(ttnn_op_full PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_full
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_full
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/full
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_full LIBRARY COMPONENT tar)
