add_library(ttnn_op_experimental_conv3d ${LIB_TYPE})
add_library(TT::NN::Ops::Experimental::Conv3d ALIAS ttnn_op_experimental_conv3d)

target_precompile_headers(ttnn_op_experimental_conv3d REUSE_FROM TT::CommonPCH)

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
        TT::NN::Core
)

install(TARGETS ttnn_op_experimental_conv3d LIBRARY COMPONENT tar)
