include(sources.cmake)

add_library(ttnn_op_experimental_transformer ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Transformer ALIAS ttnn_op_experimental_transformer)

tt_reuse_precompile_headers(ttnn_op_experimental_transformer TTNN::PCHFull)
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
    all_reduce_create_qkv_heads/device/kernels/*.cpp
    concatenate_heads/device/kernels/*.cpp
    create_qkv_heads/device/kernels/*.cpp
    create_qkv_heads_from_separate_tensors/device/kernels/*.cpp
    fused_distributed_rmsnorm/device/kernels/*.cpp
    nlp_concat_heads/device/kernels/*.cpp
    nlp_concat_heads_boltz/device/kernels/*.cpp
    nlp_concat_heads_decode/device/kernels/*.cpp
    nlp_create_qkv_heads/device/kernels/*.cpp
    nlp_create_qkv_heads_boltz/device/kernels/*.cpp
    nlp_create_qkv_heads_decode/device/kernels/*.cpp
    nlp_create_qkv_heads_falcon7b/device/kernels/*.cpp
    nlp_create_qkv_heads_segformer/device/kernels/*.cpp
    nlp_create_qkv_heads_vit/device/kernels/*.cpp
    nlp_kv_cache_load_slice/device/kernels/*.cpp
    rotary_embedding/device/kernels/*.cpp
    rotary_embedding_hf/device/kernels/*.cpp
    rotary_embedding_llama/device/kernels/*.cpp
    rotary_embedding_llama_fused_qk/device/kernels/*.cpp
    rotate_half/device/kernels/*.cpp
    split_query_key_value_and_split_heads/device/kernels/*.cpp
    dit_layernorm_pre_all_gather/device/kernels/*.cpp
    dit_layernorm_post_all_gather/device/kernels/*.cpp
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

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_EXPERIMENTAL_TRANSFORMER_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_TRANSFORMER_NANOBIND_SRCS})
endif()
