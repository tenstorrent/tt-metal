add_library(ttnn_op_experimental_maxpool3d ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::MaxPool3d ALIAS ttnn_op_experimental_maxpool3d)

target_precompile_headers(ttnn_op_experimental_maxpool3d REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_maxpool3d)

target_sources(
    ttnn_op_experimental_maxpool3d
    PRIVATE
        maxpool3d.cpp
        device/maxpool3d_device_operation.cpp
        device/maxpool3d_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_maxpool3d PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_maxpool3d
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_maxpool3d LIBRARY COMPONENT tar)
