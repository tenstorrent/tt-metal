add_library(ttnn_op_transformer ${LIB_TYPE})
add_library(TTNN::Ops::Transformer ALIAS ttnn_op_transformer)

target_precompile_headers(ttnn_op_transformer REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_transformer)

target_sources(
    ttnn_op_transformer
    PRIVATE
        attention_softmax/attention_softmax.cpp
        concatenate_heads/concatenate_heads.cpp
        sdpa/device/joint_sdpa_op.cpp
        sdpa/device/joint_sdpa_program_factory.cpp
        sdpa/device/sdpa_op.cpp
        sdpa/device/sdpa_program_factory.cpp
        sdpa/sdpa.cpp
        sdpa_decode/device/sdpa_decode_op.cpp
        sdpa_decode/device/sdpa_decode_program_factory.cpp
        sdpa_decode/sdpa_decode.cpp
        split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.cpp
)

target_include_directories(ttnn_op_transformer PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_transformer
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_transformer LIBRARY COMPONENT tar)
