add_library(ttnn_op_sliding_window ${LIB_TYPE})
add_library(TT::NN::Ops::SlidingWindow ALIAS ttnn_op_sliding_window)

target_precompile_headers(ttnn_op_sliding_window REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_sliding_window
    PRIVATE
        halo/device/halo_device_operation.cpp
        halo/device/untilize_with_halo_program_factory.cpp
        halo/halo.cpp
        sliding_window.cpp
)

target_include_directories(ttnn_op_sliding_window PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_sliding_window
    PRIVATE
        TT::Metalium
        TT::NN::Core
)

install(TARGETS ttnn_op_sliding_window LIBRARY COMPONENT tar)
