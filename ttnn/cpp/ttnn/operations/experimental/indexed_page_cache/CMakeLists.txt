include(sources.cmake)

add_library(ttnn_op_experimental_indexed_page_cache ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::IndexedPageCache ALIAS ttnn_op_experimental_indexed_page_cache)

tt_reuse_precompile_headers(ttnn_op_experimental_indexed_page_cache TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_indexed_page_cache)

set_target_properties(
    ttnn_op_experimental_indexed_page_cache
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*.cpp)

target_sources(
    ttnn_op_experimental_indexed_page_cache
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_INDEXED_PAGE_CACHE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_INDEXED_PAGE_CACHE_SRCS}
)

target_link_libraries(ttnn_op_experimental_indexed_page_cache PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_experimental_indexed_page_cache
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/indexed_page_cache
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_indexed_page_cache FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# The Python extension is shared, but this operation owns its nanobind source
# list locally so routine changes do not touch ttnn/sources.cmake.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_INDEXED_PAGE_CACHE_NANOBIND_SRCS})
endif()
