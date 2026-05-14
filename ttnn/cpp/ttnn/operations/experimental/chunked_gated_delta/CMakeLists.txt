include(sources.cmake)

add_library(ttnn_op_experimental_chunked_gated_delta ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::ChunkedGatedDelta ALIAS ttnn_op_experimental_chunked_gated_delta)

tt_reuse_precompile_headers(ttnn_op_experimental_chunked_gated_delta TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_chunked_gated_delta)

set_target_properties(
    ttnn_op_experimental_chunked_gated_delta
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_chunked_gated_delta
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
        ${TTNN_OP_EXPERIMENTAL_CHUNKED_GATED_DELTA_SRCS}
)

target_include_directories(ttnn_op_experimental_chunked_gated_delta PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_chunked_gated_delta
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_chunked_gated_delta
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/chunked_gated_delta
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_chunked_gated_delta LIBRARY COMPONENT tar)
