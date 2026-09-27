# Source files for ttnn_op_bringup_rms_norm_ttnn (the C++ host side of ttnn.bringup.rms_norm).
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BRINGUP_RMS_NORM_TTNN_API_HEADERS rms_norm_ttnn.hpp)

set(TTNN_OP_BRINGUP_RMS_NORM_TTNN_SRCS
    device/rms_norm_ttnn_device_operation.cpp
    device/rms_norm_ttnn_program_factory.cpp
    rms_norm_ttnn.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/ttnn/bringup/rms_norm_ttnn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_BRINGUP_RMS_NORM_TTNN_NANOBIND_SRCS rms_norm_ttnn_nanobind.cpp)
