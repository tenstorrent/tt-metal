# Source files for ttnn_op_pool.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_POOL_SRCS
    generic/device/pool_multi_core_program_factory.cpp
    generic/device/pool_op.cpp
    generic/generic_pools.cpp
    global_avg_pool/global_avg_pool.cpp
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
    upsample/device//upsample_bilinear_program_factory_multicore.cpp
    upsample/device/upsample_common.cpp
    upsample/device/upsample_device_operation.cpp
    upsample/device/upsample_program_factory_multicore_sharded.cpp
    upsample/device/upsample_program_factory_multicore_interleaved.cpp
    upsample/device/upsample_nearest_float_program_factory.cpp
    upsample/upsample.cpp
)
