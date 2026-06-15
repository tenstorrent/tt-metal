include(sources.cmake)

add_library(ttnn_op_prefetcher ${LIB_TYPE})
add_library(TTNN::Ops::Prefetcher ALIAS ttnn_op_prefetcher)

tt_reuse_precompile_headers(ttnn_op_prefetcher TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_prefetcher)

set_target_properties(
    ttnn_op_prefetcher
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels prefetcher/device/kernels/*)

target_sources(
    ttnn_op_prefetcher
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_PREFETCHER_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_PREFETCHER_SRCS}
)

target_link_libraries(ttnn_op_prefetcher PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_prefetcher
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/prefetcher
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_prefetcher FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)
