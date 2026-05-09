add_library(ttnn_op_experimental_fused_paged_update_and_mla ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::FusedPagedUpdateAndMla ALIAS ttnn_op_experimental_fused_paged_update_and_mla)

target_precompile_headers(ttnn_op_experimental_fused_paged_update_and_mla REUSE_FROM TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_fused_paged_update_and_mla)

set_target_properties(
    ttnn_op_experimental_fused_paged_update_and_mla
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_fused_paged_update_and_mla
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        fused_paged_update_and_mla.cpp
)

target_include_directories(ttnn_op_experimental_fused_paged_update_and_mla PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_fused_paged_update_and_mla
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_fused_paged_update_and_mla LIBRARY COMPONENT tar)
