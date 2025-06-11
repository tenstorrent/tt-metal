add_library(ttnn_op_pool ${LIB_TYPE})
add_library(TTNN::Ops::Pool ALIAS ttnn_op_pool)

target_precompile_headers(ttnn_op_pool REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_pool)

target_sources(
    ttnn_op_pool
    PRIVATE
        downsample/device/downsample_op.cpp
        downsample/device/downsample_program_factory.cpp
        downsample/downsample.cpp
        generic/device/pool_multi_core_program_factory.cpp
        generic/device/pool_op.cpp
        generic/generic_pools.cpp
        global_avg_pool/global_avg_pool.cpp
        pool_utils.cpp
        upsample/device//upsample_bilinear_program_factory_multicore.cpp
        upsample/device/upsample_op.cpp
        upsample/device/upsample_program_factory_multicore.cpp
        upsample/device/upsample_program_factory_singlecore.cpp
        upsample/upsample.cpp
)

target_include_directories(ttnn_op_pool PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_pool
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_pool LIBRARY COMPONENT tar)
