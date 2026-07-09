include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::ZeroPaddedKvCache
    ALIAS ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ZERO_PADDED_KV_CACHE_API_HEADERS}
    PRIVATE
        device/zero_padded_kv_cache_device_operation.cpp
        zero_padded_kv_cache.cpp
)

target_link_libraries(ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(TARGETS ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache LIBRARY COMPONENT tar)

install(TARGETS ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache FILE_SET api COMPONENT ttnn-dev)
