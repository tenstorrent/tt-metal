add_library(ttnn_op_conv ${LIB_TYPE})
add_library(TTNN::Ops::Conv ALIAS ttnn_op_conv)

target_precompile_headers(ttnn_op_conv REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_conv)
set_target_properties(
    ttnn_op_conv
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels conv2d/device/kernels/*)
target_sources(
    ttnn_op_conv
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        conv2d/conv2d.cpp
        conv2d/conv2d_utils.cpp
        conv2d/conv2d_op_program_factory_common.cpp
        conv2d/device/conv2d_op.cpp
        conv2d/device/conv2d_op_sharded_program_factory.cpp
        conv2d/device/conv2d_op_width_sharded_program_factory.cpp
        conv_transpose2d/conv_transpose2d.cpp
        conv_transpose2d/prepare_conv_transpose2d_weights.cpp
        conv2d/prepare_conv2d_weights.cpp
        conv1d/conv1d.cpp
)

target_include_directories(ttnn_op_conv PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_conv
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_conv
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/conv
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_conv LIBRARY COMPONENT tar)
