add_library(ttnn_op_experimental_gdn_fused ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::GdnFused ALIAS ttnn_op_experimental_gdn_fused)

tt_reuse_precompile_headers(ttnn_op_experimental_gdn_fused TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_gdn_fused)

set_target_properties(
    ttnn_op_experimental_gdn_fused
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_gdn_fused
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        device/gdn_fused_device_operation.cpp
        device/gdn_fused_program_factory.cpp
        gdn_fused.cpp
)

target_include_directories(ttnn_op_experimental_gdn_fused PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_gdn_fused
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_gdn_fused LIBRARY COMPONENT tar)
