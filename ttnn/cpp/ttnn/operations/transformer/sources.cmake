# Source files for ttnn_op_transformer.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_TRANSFORMER_SRCS
    attention_softmax/attention_softmax.cpp
    concatenate_heads/concatenate_heads.cpp
    sdpa/device/ring_fusion.cpp
    sdpa/device/joint_sdpa_device_operation.cpp
    sdpa/device/joint_sdpa_program_factory.cpp
    sdpa/device/ring_joint_sdpa_device_operation.cpp
    sdpa/device/ring_joint_sdpa_program_factory.cpp
    sdpa/device/sliding_halo_layout.cpp
    sdpa/device/exp_ring_joint_sdpa_device_operation.cpp
    sdpa/device/exp_ring_joint_sdpa_program_factory.cpp
    sdpa/device/ring_distributed_sdpa_device_operation.cpp
    sdpa/device/ring_distributed_sdpa_program_factory.cpp
    sdpa/device/sdpa_device_operation.cpp
    sdpa/device/sdpa_perf_model.cpp
    sdpa/device/sdpa_program_factory.cpp
    sdpa/sdpa.cpp
    sdpa/device/sparse_sdpa_device_operation.cpp
    sdpa/device/sparse_sdpa_program_factory.cpp
    sdpa/sparse_sdpa.cpp
    sdpa/device/sparse_sdpa_msa_device_operation.cpp
    sdpa/device/sparse_sdpa_msa_program_factory.cpp
    sdpa/sparse_sdpa_msa.cpp
    sdpa_decode/device/sdpa_decode_device_operation.cpp
    sdpa_decode/device/sdpa_decode_program_factory.cpp
    sdpa_decode/sdpa_decode.cpp
    split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.cpp
    gated_delta_attn/device/gated_delta_attn_device_operation.cpp
    gated_delta_attn/device/gated_delta_attn_program_factory.cpp
    gated_delta_attn/gated_delta_attn.cpp
    chunk_gated_delta_rule/device/chunk_gated_delta_rule_device_operation.cpp
    chunk_gated_delta_rule/device/chunk_gated_delta_rule_program_factory.cpp
    chunk_gated_delta_rule/device/chunk_gdn_phased.cpp
    chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp
    chunk_gated_delta_rule/chunk_gated_delta_rule.cpp
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
    chunk_gated_delta_rule/chunk_gated_delta_rule_nanobind.cpp
    transformer_nanobind.cpp
)
