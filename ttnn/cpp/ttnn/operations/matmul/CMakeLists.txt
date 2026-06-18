include(sources.cmake)

add_library(ttnn_op_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Matmul ALIAS ttnn_op_matmul)

tt_reuse_precompile_headers(ttnn_op_matmul TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_matmul)
set_target_properties(
    ttnn_op_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_MATMUL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_MATMUL_SRCS}
)

target_include_directories(ttnn_op_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_matmul PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_matmul
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/matmul
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_matmul LIBRARY COMPONENT tar)

# Quasar (_qsr) factory copies are byte-near-identical clones of the originals; exclude them
# from unity build so duplicated file-local helpers don't collide in a shared translation unit.
set_source_files_properties(
    device/factory/matmul_multicore_program_factory_qsr.cpp
    device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory_qsr.cpp
    device/factory/matmul_multicore_reuse_mcast_1d_program_factory_qsr.cpp
    device/factory/matmul_multicore_reuse_mcast_2d_program_factory_qsr.cpp
    device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory_qsr.cpp
    device/factory/matmul_multicore_reuse_optimized_program_factory_qsr.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)
