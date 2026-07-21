# Source files for ttnn_op_pool.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_POOL_API_HEADERS
    grid_sample/grid_sample.hpp
    grid_sample/grid_sample_prepare_grid.hpp
    rotate/rotate.hpp
)

set(TTNN_OP_POOL_SRCS
    grid_sample/device/grid_sample_device_operation.cpp
    grid_sample/device/grid_sample_bilinear_program_factory.cpp
    grid_sample/device/grid_sample_nearest_program_factory.cpp
    grid_sample/device/grid_sample_utils.cpp
    grid_sample/grid_sample.cpp
    grid_sample/grid_sample_prepare_grid.cpp
    rotate/device/rotate_device_operation.cpp
    rotate/device/rotate_nearest_program_factory.cpp
    rotate/device/rotate_bilinear_program_factory.cpp
    rotate/rotate.cpp
    pool_utils.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/pool/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_POOL_NANOBIND_SRCS
    grid_sample/grid_sample_nanobind.cpp
    rotate/rotate_nanobind.cpp
)
