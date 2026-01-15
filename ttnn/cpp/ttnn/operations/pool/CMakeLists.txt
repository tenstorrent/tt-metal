add_library(ttnn_op_pool ${LIB_TYPE})
add_library(TTNN::Ops::Pool ALIAS ttnn_op_pool)

target_precompile_headers(ttnn_op_pool REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_pool)

set_target_properties(
    ttnn_op_pool
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    generic/device/kernels/*
    grid_sample/device/kernels/*
    upsample/device/kernels/*
)

target_sources(
    ttnn_op_pool
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
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
        pool_utils.cpp
        upsample/device//upsample_bilinear_program_factory_multicore.cpp
        upsample/device/upsample_common.cpp
        upsample/device/upsample_device_operation.cpp
        upsample/device/upsample_program_factory_multicore_sharded.cpp
        upsample/device/upsample_program_factory_multicore_interleaved.cpp
        upsample/device/upsample_nearest_float_program_factory.cpp
        upsample/upsample.cpp
)

target_include_directories(ttnn_op_pool PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_pool
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_pool
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/pool
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_pool LIBRARY COMPONENT tar)
