add_library(ttnn_op_debug ${LIB_TYPE})
add_library(TTNN::Ops::Debug ALIAS ttnn_op_debug)

target_precompile_headers(ttnn_op_debug REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_debug)

file(GLOB_RECURSE kernels kernels/*)

target_sources(
    ttnn_op_debug
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        apply_device_delay.cpp
        device/apply_device_delay_device_operation.cpp
)

target_include_directories(ttnn_op_debug PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_debug
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_debug
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/debug
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_debug LIBRARY COMPONENT tar)
