add_library(ttnn_op_prefetcher ${LIB_TYPE})
add_library(TTNN::Ops::Prefetcher ALIAS ttnn_op_prefetcher)

target_precompile_headers(ttnn_op_prefetcher REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_prefetcher)

file(GLOB_RECURSE kernels prefetcher/device/kernels/*)

target_sources(
    ttnn_op_prefetcher
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        prefetcher/device/dram_prefetcher_device_operation.cpp
        prefetcher/device/dram_prefetcher_program_factory.cpp
        prefetcher/dram_prefetcher.cpp
)

target_include_directories(ttnn_op_prefetcher PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_prefetcher
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_prefetcher
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/prefetcher
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_prefetcher LIBRARY COMPONENT tar)
