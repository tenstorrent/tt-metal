include(sources.cmake)

add_library(ttnn_op_kv_sdpa ${LIB_TYPE})
add_library(TTNN::Ops::KvSdpa ALIAS ttnn_op_kv_sdpa)

tt_reuse_precompile_headers(ttnn_op_kv_sdpa TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_kv_sdpa)

set_target_properties(
    ttnn_op_kv_sdpa
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_kv_sdpa
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
    PRIVATE
        ${TTNN_OP_KV_SDPA_SRCS}
)

target_include_directories(ttnn_op_kv_sdpa PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_kv_sdpa PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_kv_sdpa LIBRARY COMPONENT tar)
