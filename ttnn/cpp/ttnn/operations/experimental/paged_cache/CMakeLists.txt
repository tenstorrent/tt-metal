add_library(ttnn_op_experimental_paged_cache ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::PagedCache ALIAS ttnn_op_experimental_paged_cache)

target_precompile_headers(ttnn_op_experimental_paged_cache REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_paged_cache)

set_target_properties(
    ttnn_op_experimental_paged_cache
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_paged_cache
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
        device/update_cache/paged_update_cache_device_operation.cpp
        device/update_cache/paged_update_cache_program_factory.cpp
        device/fused_update_cache/paged_fused_update_cache_device_operation.cpp
        device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp
        device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp
        device/fill_cache/paged_fill_cache_device_operation.cpp
        device/fill_cache/paged_fill_cache_program_factory.cpp
        paged_cache.cpp
)

target_include_directories(ttnn_op_experimental_paged_cache PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_paged_cache
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_paged_cache
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/paged_cache
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_paged_cache LIBRARY COMPONENT tar)
