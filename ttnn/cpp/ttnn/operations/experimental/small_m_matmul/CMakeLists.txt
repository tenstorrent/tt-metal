include(sources.cmake)

add_library(ttnn_op_experimental_small_m_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::SmallMMatmul ALIAS ttnn_op_experimental_small_m_matmul)

tt_reuse_precompile_headers(ttnn_op_experimental_small_m_matmul TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_small_m_matmul)

set_target_properties(
    ttnn_op_experimental_small_m_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_small_m_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_SMALL_M_MATMUL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_SMALL_M_MATMUL_SRCS}
)

# Disable unity build for program factory files to avoid symbol collisions
set_source_files_properties(
    device/small_m_matmul_program_factory.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

target_link_libraries(ttnn_op_experimental_small_m_matmul PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_small_m_matmul
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/small_m_matmul
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_small_m_matmul FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings are registered on the shared `ttnn` module target from here (main's per-op pattern), not
# centrally in ttnn/sources.cmake. The nanobind source is kept out of unity batching because neighbouring
# pybind sources rely on unity chunk-order namespace visibility.
if(TARGET ttnn)
    set_source_files_properties(
        small_m_matmul_nanobind.cpp
        PROPERTIES
            SKIP_UNITY_BUILD_INCLUSION
                ON
    )
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_SMALL_M_MATMUL_NANOBIND_SRCS})
endif()
