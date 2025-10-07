add_library(ttnn_op_experimental_transformer ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Transformer ALIAS ttnn_op_experimental_transformer)

target_precompile_headers(ttnn_op_experimental_transformer REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_transformer)
set_target_properties(
    ttnn_op_experimental_transformer
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE nlp_kv_cache_load_slice/device/kernels/*)
target_sources(
    ttnn_op_experimental_transformer
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES nlp_kv_cache_load_slice/nlp_kv_cache_load_slice.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        all_reduce_create_qkv_heads/all_reduce_create_qkv_heads.cpp
        all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_op.cpp
        all_reduce_create_qkv_heads/device/all_reduce_create_qkv_heads_program_factory.cpp
        concatenate_heads/device/concatenate_heads_device_operation.cpp
        create_qkv_heads/create_qkv_heads.cpp
        create_qkv_heads/device/create_qkv_heads_device_operation.cpp
        create_qkv_heads/device/create_qkv_heads_program_factory.cpp
        create_qkv_heads_from_separate_tensors/create_qkv_heads_from_separate_tensors.cpp
        create_qkv_heads_from_separate_tensors/device/create_qkv_heads_from_separate_tensors_device_operation.cpp
        create_qkv_heads_from_separate_tensors/device/create_qkv_heads_from_separate_tensors_program_factory.cpp
        nlp_concat_heads/device/nlp_concat_heads_device_operation.cpp
        nlp_concat_heads/device/nlp_concat_heads_program_factory.cpp
        nlp_concat_heads/nlp_concat_heads.cpp
        nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp
        nlp_concat_heads_decode/device/nlp_concat_heads_decode_program_factory.cpp
        nlp_concat_heads_decode/nlp_concat_heads_decode.cpp
        nlp_concat_heads_boltz/device/nlp_concat_heads_boltz_device_operation.cpp
        nlp_concat_heads_boltz/device/nlp_concat_heads_boltz_program_factory.cpp
        nlp_concat_heads_boltz/nlp_concat_heads_boltz.cpp
        nlp_create_qkv_heads/device/nlp_create_qkv_heads_device_operation.cpp
        nlp_create_qkv_heads/device/nlp_create_qkv_heads_program_factory.cpp
        nlp_create_qkv_heads/nlp_create_qkv_heads.cpp
        nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.cpp
        nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_program_factory.cpp
        nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.cpp
        nlp_create_qkv_heads_falcon7b/device/nlp_create_qkv_heads_falcon7b_device_operation.cpp
        nlp_create_qkv_heads_falcon7b/device/nlp_create_qkv_heads_falcon7b_program_factory.cpp
        nlp_create_qkv_heads_falcon7b/nlp_create_qkv_heads_falcon7b.cpp
        nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_device_operation.cpp
        nlp_create_qkv_heads_segformer/device/nlp_create_qkv_heads_segformer_program_factory.cpp
        nlp_create_qkv_heads_segformer/nlp_create_qkv_heads_segformer.cpp
        nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_device_operation.cpp
        nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_program_factory.cpp
        nlp_create_qkv_heads_vit/nlp_create_qkv_heads_vit.cpp
        nlp_create_qkv_heads_boltz/device/nlp_create_qkv_heads_boltz_device_operation.cpp
        nlp_create_qkv_heads_boltz/device/nlp_create_qkv_heads_boltz_program_factory.cpp
        nlp_create_qkv_heads_boltz/nlp_create_qkv_heads_boltz.cpp
        nlp_kv_cache_load_slice/device/nlp_kv_cache_load_slice_device_operation.cpp
        nlp_kv_cache_load_slice/device/nlp_kv_cache_load_slice_program_factory.cpp
        nlp_kv_cache_load_slice/nlp_kv_cache_load_slice.cpp
        rotary_embedding/device/rotary_embedding_device_operation.cpp
        rotary_embedding/device/rotary_embedding_program_factory.cpp
        rotary_embedding/rotary_embedding.cpp
        rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp
        rotary_embedding_llama/device/rotary_embedding_llama_program_factory.cpp
        rotary_embedding_llama/rotary_embedding_llama.cpp
        rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation.cpp
        rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_program_factory.cpp
        rotary_embedding_llama_fused_qk/rotary_embedding_llama_fused_qk.cpp
        rotate_half/device/rotate_half_device_operation.cpp
        rotate_half/device/single_core/rotate_half_program_factory.cpp
        rotate_half/rotate_half.cpp
        split_query_key_value_and_split_heads/device/split_query_key_value_and_split_heads_device_operation.cpp
)

target_include_directories(ttnn_op_experimental_transformer PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_transformer
    PRIVATE
        TT::Metalium
        TTNN::Core
)

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
