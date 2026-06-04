# Source files for ttnn_op_kv_cache.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_KV_CACHE_SRCS
    device/update_cache_device_operation.cpp
    device/fill_cache_multi_core_program_factory.cpp
    device/update_cache_multi_core_program_factory.cpp
    device/zero_cache_range_device_operation.cpp
    device/zero_cache_range_program_factory.cpp
    kv_cache.cpp
)

set(TTNN_OP_KV_CACHE_API_HEADERS kv_cache.hpp)
