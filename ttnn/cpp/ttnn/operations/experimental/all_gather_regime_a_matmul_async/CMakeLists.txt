include(sources.cmake)

add_library(ttnn_op_experimental_all_gather_regime_a_matmul_async ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::AllGatherRegimeAMatmulAsync
    ALIAS ttnn_op_experimental_all_gather_regime_a_matmul_async
)

tt_reuse_precompile_headers(ttnn_op_experimental_all_gather_regime_a_matmul_async TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_all_gather_regime_a_matmul_async)

set_target_properties(
    ttnn_op_experimental_all_gather_regime_a_matmul_async
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_all_gather_regime_a_matmul_async
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_SRCS}
)

# Disable unity build for program factory files to avoid symbol collisions
set_source_files_properties(
    device/all_gather_regime_a_matmul_async_program_factory.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

# Links the single-chip Regime-A op for the shared RegimeAMatmulConfig / auto_select_config picker and the
# host planner (regime_a_matmul_plan.hpp). Those are shared verbatim rather than duplicated.
# CCL provides all_gather_async, which the Phase-0 (unfused) composition calls to materialise the gathered
# activation. When the fused path lands this dependency should disappear from the hot path, but it stays
# for the A/B baseline.
target_link_libraries(
    ttnn_op_experimental_all_gather_regime_a_matmul_async
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
        TTNN::Ops::Experimental::RegimeAMatmul
        TTNN::Ops::Experimental::CCL
)

install(
    TARGETS
        ttnn_op_experimental_all_gather_regime_a_matmul_async
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/all_gather_regime_a_matmul_async
        COMPONENT ttnn-runtime
)

install(
    TARGETS
        ttnn_op_experimental_all_gather_regime_a_matmul_async
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

# Python bindings are registered on the shared `ttnn` module target from here (main's per-op pattern), not
# centrally in ttnn/sources.cmake. The nanobind source is kept out of unity batching because neighbouring
# pybind sources rely on unity chunk-order namespace visibility.
if(TARGET ttnn)
    set_source_files_properties(
        all_gather_regime_a_matmul_async_nanobind.cpp
        PROPERTIES
            SKIP_UNITY_BUILD_INCLUSION
                ON
    )
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_NANOBIND_SRCS})
endif()
