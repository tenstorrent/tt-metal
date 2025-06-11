add_library(ttnn_op_experimental_cnn OBJECT)
add_library(TTNN::Ops::Experimental::CNN ALIAS ttnn_op_experimental_cnn)

target_precompile_headers(ttnn_op_experimental_cnn REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_cnn)

target_sources(
    ttnn_op_experimental_cnn
    PRIVATE
        convert_to_chw/convert_to_chw.cpp
        convert_to_chw/device/convert_to_chw_op.cpp
        convert_to_chw/device/convert_to_chw_program_factory.cpp
        convert_to_hwc/convert_to_hwc.cpp
        convert_to_hwc/device/convert_to_hwc_op.cpp
        convert_to_hwc/device/convert_to_hwc_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_cnn PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_cnn
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_cnn LIBRARY COMPONENT tar)
