# Source files for ttnn_op_pool.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_POOL_API_HEADERS)

set(TTNN_OP_POOL_SRCS
    # TODO(nuked-op pool): generic/grid_sample/rotate/upsample sources removed for eval.
    # Shared device/kernels/ headers are intentionally retained (see CMakeLists.txt glob).
    # Placeholder keeps ttnn_op_pool non-empty; restore real sources on recreate.
    pool_placeholder.cpp
)
