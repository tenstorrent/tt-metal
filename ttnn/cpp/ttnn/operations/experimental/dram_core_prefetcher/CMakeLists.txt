include(sources.cmake)

add_library(ttnn_op_experimental_dram_core_prefetcher ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DramCorePrefetcher ALIAS ttnn_op_experimental_dram_core_prefetcher)

tt_reuse_precompile_headers(ttnn_op_experimental_dram_core_prefetcher TTNN::PCHLean)

set_target_properties(
    ttnn_op_experimental_dram_core_prefetcher
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_dram_core_prefetcher
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DRAM_CORE_PREFETCHER_SRCS}
)

target_include_directories(ttnn_op_experimental_dram_core_prefetcher PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_dram_core_prefetcher
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_dram_core_prefetcher LIBRARY COMPONENT tar)
