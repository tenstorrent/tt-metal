# Source files for ttnn_op_kv_cache.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_KV_CACHE_SRCS
    device/update_cache_device_operation.cpp
    device/fill_cache_multi_core_program_factory.cpp
    device/update_cache_multi_core_program_factory.cpp
    kv_cache.cpp
)

set(TTNN_OP_KV_CACHE_API_HEADERS kv_cache.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/kv_cache/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_KV_CACHE_NANOBIND_SRCS kv_cache_nanobind.cpp)
