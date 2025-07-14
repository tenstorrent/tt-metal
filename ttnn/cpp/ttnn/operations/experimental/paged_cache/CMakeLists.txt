add_library(ttnn_op_experimental_paged_cache ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::PagedCache ALIAS ttnn_op_experimental_paged_cache)

target_precompile_headers(ttnn_op_experimental_paged_cache REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_paged_cache)

target_sources(
    ttnn_op_experimental_paged_cache
    PRIVATE
        device/paged_cache_operation.cpp
        device/paged_fill_cache_program_factory.cpp
        device/paged_fused_update_cache_program_factory.cpp
        device/paged_update_cache_program_factory.cpp
        paged_cache.cpp
)

target_include_directories(ttnn_op_experimental_paged_cache PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_paged_cache
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_paged_cache LIBRARY COMPONENT tar)
