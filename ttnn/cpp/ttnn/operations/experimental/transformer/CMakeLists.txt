include(sources.cmake)

add_library(ttnn_op_experimental_transformer ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Transformer ALIAS ttnn_op_experimental_transformer)

tt_reuse_precompile_headers(ttnn_op_experimental_transformer TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_transformer)
set_target_properties(
    ttnn_op_experimental_transformer
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    all_reduce_create_qkv_heads/device/kernels/*
    concatenate_heads/device/kernels/*
    create_qkv_heads/device/kernels/*
    create_qkv_heads_from_separate_tensors/device/kernels/*
    fused_distributed_rmsnorm/device/kernels/*
    nlp_concat_heads/device/kernels/*
    nlp_concat_heads_boltz/device/kernels/*
    nlp_concat_heads_decode/device/kernels/*
    nlp_create_qkv_heads/device/kernels/*
    nlp_create_qkv_heads_boltz/device/kernels/*
    nlp_create_qkv_heads_decode/device/kernels/*
    nlp_create_qkv_heads_falcon7b/device/kernels/*
    nlp_create_qkv_heads_segformer/device/kernels/*
    nlp_create_qkv_heads_vit/device/kernels/*
    nlp_kv_cache_load_slice/device/kernels/*
    rotary_embedding/device/kernels/*
    rotary_embedding_llama/device/kernels/*
    rotary_embedding_llama_fused_qk/device/kernels/*
    rotate_half/device/kernels/*
    split_query_key_value_and_split_heads/device/kernels/*
    dit_layernorm_pre_all_gather/device/kernels/*
    dit_layernorm_post_all_gather/device/kernels/*
)
target_sources(
    ttnn_op_experimental_transformer
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_EXPERIMENTAL_TRANSFORMER_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_TRANSFORMER_SRCS}
)

target_include_directories(ttnn_op_experimental_transformer PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_transformer PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_experimental_transformer
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/transformer
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_transformer LIBRARY COMPONENT tar)
