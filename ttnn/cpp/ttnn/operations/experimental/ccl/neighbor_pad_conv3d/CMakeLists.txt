add_library(ttnn_op_experimental_neighbor_pad_conv3d ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::NeighborPadConv3d ALIAS ttnn_op_experimental_neighbor_pad_conv3d)

tt_reuse_precompile_headers(ttnn_op_experimental_neighbor_pad_conv3d TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_neighbor_pad_conv3d)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_neighbor_pad_conv3d
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        neighbor_pad_conv3d.cpp
        device/neighbor_pad_conv3d_device_operation.cpp
        device/neighbor_pad_conv3d_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_neighbor_pad_conv3d PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_neighbor_pad_conv3d
    PRIVATE
        TT::Metalium
        TTNN::Core
        TTNN::Ops::Experimental::CCL
        TTNN::Ops::Experimental::Conv3d
)

install(
    TARGETS
        ttnn_op_experimental_neighbor_pad_conv3d
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_conv3d
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_neighbor_pad_conv3d LIBRARY COMPONENT tar)
