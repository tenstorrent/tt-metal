# Source files for ttnn_op_experimental_indexed_page_cache.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_INDEXED_PAGE_CACHE_API_HEADERS indexed_page_cache.hpp)

set(TTNN_OP_EXPERIMENTAL_INDEXED_PAGE_CACHE_SRCS
    device/indexed_fused_update_cache/indexed_fused_update_cache_device_operation.cpp
    device/indexed_fused_update_cache/indexed_fused_update_cache_program_factory.cpp
    indexed_page_cache.cpp
)

set(TTNN_OP_EXPERIMENTAL_INDEXED_PAGE_CACHE_NANOBIND_SRCS indexed_page_cache_nanobind.cpp)
