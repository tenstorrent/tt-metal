# Source files for ttnn_op_normalization.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_NORMALIZATION_SRCS
    batch_norm/batch_norm.cpp
    batch_norm/device/batch_norm_device_operation.cpp
    batch_norm/device/batch_norm_program_factory.cpp
    batch_norm/device/running_statistics_device_operation.cpp
    batch_norm/device/running_statistics_program_factory.cpp
    batch_norm/device/batch_norm_utils.cpp
    shard_spec_validation.cpp
)

set(TTNN_OP_NORMALIZATION_API_HEADERS
)
