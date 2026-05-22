# Source files for ttnn_op_experimental_paged_cache.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_PAGED_CACHE_SRCS
    device/update_cache/paged_update_cache_device_operation.cpp
    device/update_cache/paged_update_cache_program_factory.cpp
    device/fused_update_cache/paged_fused_update_cache_device_operation.cpp
    device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp
    device/fused_update_cache/paged_row_major_fused_update_cache_program_factory.cpp
    device/fill_cache/paged_fill_cache_device_operation.cpp
    device/fill_cache/paged_fill_cache_program_factory.cpp
    paged_cache.cpp
)
