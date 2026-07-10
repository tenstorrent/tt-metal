# Source files for ttnn_op_transformer.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_TRANSFORMER_SRCS
    concatenate_heads/concatenate_heads.cpp
    split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.cpp
    gated_delta_attn/device/gated_delta_attn_device_operation.cpp
    gated_delta_attn/device/gated_delta_attn_program_factory.cpp
    gated_delta_attn/gated_delta_attn.cpp
)
