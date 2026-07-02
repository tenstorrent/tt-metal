include(sources.cmake)

add_library(ttnn_op_index_fill ${LIB_TYPE})
add_library(TTNN::Ops::IndexFill ALIAS ttnn_op_index_fill)

tt_reuse_precompile_headers(ttnn_op_index_fill TTNN::PCHFull)
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
        FILES ${TTNN_OP_INDEX_FILL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_INDEX_FILL_SRCS}
)

target_link_libraries(ttnn_op_index_fill PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_index_fill
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/index_fill
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_index_fill FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
