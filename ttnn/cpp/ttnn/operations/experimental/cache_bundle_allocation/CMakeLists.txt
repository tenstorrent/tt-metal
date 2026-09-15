include(sources.cmake)

add_library(ttnn_op_experimental_cache_bundle_allocation ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::CacheBundleAllocation ALIAS ttnn_op_experimental_cache_bundle_allocation)

tt_reuse_precompile_headers(ttnn_op_experimental_cache_bundle_allocation TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_cache_bundle_allocation)

set_target_properties(
    ttnn_op_experimental_cache_bundle_allocation
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_cache_bundle_allocation
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_CACHE_BUNDLE_ALLOCATION_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_EXPERIMENTAL_CACHE_BUNDLE_ALLOCATION_KERNELS}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_CACHE_BUNDLE_ALLOCATION_SRCS}
)

target_link_libraries(ttnn_op_experimental_cache_bundle_allocation PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_cache_bundle_allocation
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/cache_bundle_allocation
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_cache_bundle_allocation FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# The Python extension is shared, but this operation owns its nanobind source
# list locally so routine changes do not touch ttnn/sources.cmake.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_CACHE_BUNDLE_ALLOCATION_NANOBIND_SRCS})
endif()
