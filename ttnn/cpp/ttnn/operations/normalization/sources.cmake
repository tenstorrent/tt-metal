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

set(TTNN_OP_NORMALIZATION_API_HEADERS)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_NORMALIZATION_NANOBIND_SRCS
    batch_norm/batch_norm_nanobind.cpp
    normalization_nanobind.cpp
)
