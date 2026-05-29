include(sources.cmake)

add_library(ttnn_op_conv ${LIB_TYPE})
add_library(TTNN::Ops::Conv ALIAS ttnn_op_conv)

tt_reuse_precompile_headers(ttnn_op_conv TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_conv)
set_target_properties(
    ttnn_op_conv
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels conv2d/device/kernels/*)
target_sources(
    ttnn_op_conv
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_CONV_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_CONV_SRCS}
)

target_include_directories(ttnn_op_conv PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_conv PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_conv
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/conv
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_conv FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
