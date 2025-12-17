add_library(ttnn_op_experimental_conv3d ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Conv3d ALIAS ttnn_op_experimental_conv3d)

target_precompile_headers(ttnn_op_experimental_conv3d REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_conv3d)

set_target_properties(
    ttnn_op_experimental_conv3d
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_conv3d
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

install(
    TARGETS
        ttnn_op_experimental_conv3d
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/conv3d
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_conv3d LIBRARY COMPONENT tar)
