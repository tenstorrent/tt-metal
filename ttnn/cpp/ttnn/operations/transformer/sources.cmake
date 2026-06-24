# Source files for ttnn_op_transformer.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_TRANSFORMER_SRCS
    concatenate_heads/concatenate_heads.cpp
    gated_delta_attn/device/gated_delta_attn_device_operation.cpp
    gated_delta_attn/device/gated_delta_attn_program_factory.cpp
    gated_delta_attn/gated_delta_attn.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/transformer/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_TRANSFORMER_NANOBIND_SRCS
    attention_softmax/attention_softmax_nanobind.cpp
    concatenate_heads/concatenate_heads_nanobind.cpp
    sdpa/sdpa_nanobind.cpp
    sdpa_decode/sdpa_decode_nanobind.cpp
    split_query_key_value_and_split_heads/split_query_key_value_and_split_heads_nanobind.cpp
    gated_delta_attn/gated_delta_attn_nanobind.cpp
    transformer_nanobind.cpp
)
