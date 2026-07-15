include(sources.cmake)

add_library(ttnn_op_experimental_ccl ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::CCL ALIAS ttnn_op_experimental_ccl)

tt_reuse_precompile_headers(ttnn_op_experimental_ccl TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_ccl)

set_target_properties(
    ttnn_op_experimental_ccl
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    all_gather_async/device/kernels/*.cpp
    all_gather_concat_heads_fused/device/kernels/*.cpp
    all_reduce_async/device/kernels/*.cpp
    all_to_all_async/device/kernels/*.cpp
    all_to_all_async_generic/device/kernels/*.cpp
    all_to_all_dispatch_metadata/device/kernels/*.cpp
    llama_all_gather_matmul_async/device/kernels/*.cpp
    llama_reduce_scatter/device/kernels/*.cpp
    llama_reduce_scatter_create_heads/device/kernels/*.cpp
    neighbor_pad_async/device/kernels/*.cpp
    reduce_scatter_minimal_async/device/kernels/*.cpp
    ring_attention_all_gather_async/device/kernels/*.cpp
    rms_allgather/device/kernels/*.cpp
    rms_allgather/device/kernels/*.hpp
    dit_fused_distributed_rmsnorm/device/kernels/*.cpp
    dit_fused_distributed_rmsnorm/device/kernels/*.hpp
    send_recv_async/send_async/device/kernels/*.cpp
    send_recv_async/recv_async/device/kernels/*.cpp
    send_recv_async/recv_async_h2d/device/kernels/*.cpp
    send_recv_async/send_async_d2h/device/kernels/*.cpp
    slice_reshard_async/device/kernels/*.cpp
    strided_all_gather_async/device/kernels/*.cpp
    strided_all_gather_async/device/kernels/*.hpp
    strided_reduce_scatter_async/device/kernels/*.cpp
    strided_reduce_scatter_async/device/kernels/*.hpp
    deepseek_moe_reduce_scatter/device/kernels/*.cpp
    deepseek_b1_reduce_to_one/device/kernels/*.cpp
    moe_compute/device/kernels/*.cpp
    moe_compute/device/kernels/*.h
    moe/selective_reduce_combine/device/kernels/*.cpp
)

target_sources(
    ttnn_op_experimental_ccl
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_CCL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_CCL_SRCS}
)

target_include_directories(ttnn_op_experimental_ccl PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_ccl PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_experimental_ccl LIBRARY COMPONENT tar)

install(
    TARGETS
        ttnn_op_experimental_ccl
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl
        COMPONENT ttnn-runtime
)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's own
# registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_EXPERIMENTAL_CCL_NANOBIND_SRCS) lives in
# sources.cmake, not here: this file carries metalium-developers-infra as a
# required co-owner (so infra can land sweeping build-system changes), and
# routine add/remove/rename of a nanobind source shouldn't need to go
# through infra. This wiring line itself is build logic and is expected to
# change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_CCL_NANOBIND_SRCS})
endif()
