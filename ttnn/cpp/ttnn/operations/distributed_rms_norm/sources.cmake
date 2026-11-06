# Source files for ttnn_op_distributed_rms_norm.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_DISTRIBUTED_RMS_NORM_API_HEADERS distributed_rms_norm.hpp)

set(TTNN_OP_DISTRIBUTED_RMS_NORM_SRCS
    distributed_rms_norm.cpp
    device/distributed_rms_norm_device_operation.cpp
    device/distributed_rms_norm_program_factory.cpp
    device/distributed_rms_norm_planner.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/distributed_rms_norm/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_DISTRIBUTED_RMS_NORM_NANOBIND_SRCS distributed_rms_norm_nanobind.cpp)
