add_library(ttnn_op_experimental_conv3d ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Conv3d ALIAS ttnn_op_experimental_conv3d)

if(TT_METAL_BUILD_AGGRESSIVE_PCH)
    target_precompile_headers(ttnn_op_experimental_conv3d REUSE_FROM ttnn_core)
else()
    target_precompile_headers(ttnn_op_experimental_conv3d REUSE_FROM TT::CommonPCH)
endif()
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_conv3d)

target_sources(
    ttnn_op_experimental_conv3d
    PRIVATE
        conv3d.cpp
        device/conv3d_device_operation.cpp
        device/conv3d_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_conv3d PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_conv3d
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_conv3d LIBRARY COMPONENT tar)
