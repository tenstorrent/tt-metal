# Source files for ttnn_op_experimental_paged_cache.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_PAGED_CACHE_API_HEADERS paged_cache.hpp)

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

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/paged_cache/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_PAGED_CACHE_NANOBIND_SRCS paged_cache_nanobind.cpp)
