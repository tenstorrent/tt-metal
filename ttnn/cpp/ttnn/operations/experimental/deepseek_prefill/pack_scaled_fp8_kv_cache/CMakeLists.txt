include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::PackScaledFp8KvCache
    ALIAS ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache REUSE_FROM TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PACK_SCALED_FP8_KV_CACHE_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PACK_SCALED_FP8_KV_CACHE_SRCS}
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_prefill_pack_scaled_fp8_kv_cache
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache
        COMPONENT ttnn-runtime
    FILE_SET
    api
        COMPONENT ttnn-dev
    LIBRARY
        COMPONENT tar
)

# Register this op's binding directly on the shared Python module target.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_PACK_SCALED_FP8_KV_CACHE_NANOBIND_SRCS})
endif()
