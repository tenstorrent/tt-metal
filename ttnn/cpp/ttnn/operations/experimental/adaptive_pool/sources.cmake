# Source files for ttnn_op_experimental_adaptive_pool.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_ADAPTIVE_POOL_API_HEADERS adaptive_pools.hpp)

set(TTNN_OP_EXPERIMENTAL_ADAPTIVE_POOL_SRCS
    adaptive_pools.cpp
    adaptive_pool_utils.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/adaptive_pool/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_ADAPTIVE_POOL_NANOBIND_SRCS adaptive_pools_nanobind.cpp)
