add_library(ttnn_op_conv2d_new ${LIB_TYPE})
add_library(TTNN::Ops::Conv2dNew ALIAS ttnn_op_conv2d_new)

target_precompile_headers(ttnn_op_conv2d_new REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_conv2d_new)

set_target_properties(
    ttnn_op_conv2d_new
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_conv2d_new
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        device/conv2d_new_device_operation.cpp
        device/factory/sharded_descriptor.cpp
        device/factory/width_sharded_descriptor.cpp
)

target_include_directories(ttnn_op_conv2d_new PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_conv2d_new
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::Conv # for reusing Conv2dDeviceOperation methods and utilities
)

install(TARGETS ttnn_op_conv2d_new LIBRARY COMPONENT tar)
