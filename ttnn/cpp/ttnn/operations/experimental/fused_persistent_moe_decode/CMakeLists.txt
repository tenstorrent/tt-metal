add_library(ttnn_op_experimental_fused_persistent_moe_decode ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::FusedPersistentMoeDecode ALIAS ttnn_op_experimental_fused_persistent_moe_decode)

target_precompile_headers(ttnn_op_experimental_fused_persistent_moe_decode REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_fused_persistent_moe_decode)

set_target_properties(
    ttnn_op_experimental_fused_persistent_moe_decode
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_fused_persistent_moe_decode
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        fused_persistent_moe_decode.cpp
        device/fused_persistent_moe_decode_device_operation.cpp
        device/fused_persistent_moe_decode_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_fused_persistent_moe_decode PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_fused_persistent_moe_decode
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_fused_persistent_moe_decode LIBRARY COMPONENT tar)
