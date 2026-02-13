add_library(ttnn_op_matmul_new ${LIB_TYPE})
add_library(TTNN::Ops::MatmulNew ALIAS ttnn_op_matmul_new)

target_precompile_headers(ttnn_op_matmul_new REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_matmul_new)

set_target_properties(
    ttnn_op_matmul_new
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_matmul_new
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        device/matmul_new_device_operation.cpp
        device/factory/multicore_descriptor.cpp
        device/factory/reuse_optimized_descriptor.cpp
        device/factory/reuse_mcast_1d_descriptor.cpp
        device/factory/reuse_mcast_2d_descriptor.cpp
        device/factory/dram_sharded_descriptor.cpp
        device/factory/batched_hs_dram_sharded_descriptor.cpp
)

target_include_directories(ttnn_op_matmul_new PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_matmul_new
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::Matmul # for reusing MatmulDeviceOperation methods and utilities
)

install(TARGETS ttnn_op_matmul_new LIBRARY COMPONENT tar)
