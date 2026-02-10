add_library(ttnn_op_experimental_cnn OBJECT)
add_library(TTNN::Ops::Experimental::CNN ALIAS ttnn_op_experimental_cnn)

target_precompile_headers(ttnn_op_experimental_cnn REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_cnn)

set_target_properties(
    ttnn_op_experimental_cnn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    convert_to_chw/device/kernels/*
    convert_to_hwc/device/kernels/*
)

target_sources(
    ttnn_op_experimental_cnn
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
        convert_to_chw/convert_to_chw.cpp
        convert_to_chw/device/convert_to_chw_device_operation.cpp
        convert_to_chw/device/convert_to_chw_program_factory.cpp
        convert_to_hwc/convert_to_hwc.cpp
        convert_to_hwc/device/convert_to_hwc_device_operation.cpp
        convert_to_hwc/device/convert_to_hwc_program_factory.cpp
        convert_to_hwc/device/gather.cpp
)

target_include_directories(ttnn_op_experimental_cnn PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_cnn
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_cnn
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/cnn
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_cnn LIBRARY COMPONENT tar)
