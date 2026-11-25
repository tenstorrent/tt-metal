# Source files for ttnn_op_all_gather_rms_norm.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ALL_GATHER_RMS_NORM_API_HEADERS all_gather_rms_norm.hpp)

set(TTNN_OP_ALL_GATHER_RMS_NORM_SRCS
    all_gather_rms_norm.cpp
    device/all_gather_rms_norm_device_operation.cpp
    device/all_gather_rms_norm_program_factory.cpp
    device/all_gather_rms_norm_planner.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/all_gather_rms_norm/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_ALL_GATHER_RMS_NORM_NANOBIND_SRCS all_gather_rms_norm_nanobind.cpp)
