include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeekPrefill::RotaryEmbeddingIndexed
    ALIAS ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed
)

target_precompile_headers(ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed)

set_target_properties(
    ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ROTARY_EMBEDDING_INDEXED_API_HEADERS}
    PRIVATE
        device/rotary_embedding_indexed_device_operation.cpp
        rotary_embedding_indexed.cpp
)

target_link_libraries(
    ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed
    PRIVATE
        TT::Metalium
    PUBLIC
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed LIBRARY COMPONENT tar)

install(TARGETS ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed FILE_SET api COMPONENT ttnn-dev)
