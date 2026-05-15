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
    sdpa/device/exp_ring_joint_sdpa_device_operation.cpp
    sdpa/device/exp_ring_joint_sdpa_program_factory.cpp
    sdpa/device/ring_distributed_sdpa_device_operation.cpp
    sdpa/device/ring_distributed_sdpa_program_factory.cpp
    sdpa/device/sdpa_device_operation.cpp
    sdpa/device/sdpa_perf_model.cpp
    sdpa/device/sdpa_program_factory.cpp
    sdpa/sdpa.cpp
    sdpa_decode/device/sdpa_decode_device_operation.cpp
    sdpa_decode/device/sdpa_decode_program_factory.cpp
    sdpa_decode/sdpa_decode.cpp
    sdpa_windowed/device/sdpa_windowed_device_operation.cpp
    sdpa_windowed/device/sdpa_windowed_program_factory.cpp
    sdpa_windowed/sdpa_windowed.cpp
    split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.cpp
)
