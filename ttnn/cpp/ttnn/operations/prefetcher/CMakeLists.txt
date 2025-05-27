add_library(ttnn_op_prefetcher ${LIB_TYPE})
add_library(TT::NN::Ops::Prefetcher ALIAS ttnn_op_prefetcher)

target_precompile_headers(ttnn_op_prefetcher REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_prefetcher
    PRIVATE
        prefetcher/device/dram_prefetcher_op.cpp
        prefetcher/device/dram_prefetcher_op_multi_core.cpp
        prefetcher/dram_prefetcher.cpp
)

target_include_directories(ttnn_op_prefetcher PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_prefetcher
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_prefetcher LIBRARY COMPONENT tar)
