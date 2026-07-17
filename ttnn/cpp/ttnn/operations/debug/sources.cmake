# Source files for ttnn_op_debug.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_DEBUG_SRCS
    apply_device_delay.cpp
    device/apply_device_delay_device_operation.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/debug/CMakeLists.txt (see the `if(TARGET ttnn)`
# block there). Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra as a
# required co-owner.
set(TTNN_OP_DEBUG_NANOBIND_SRCS
    debug_nanobind.cpp
    apply_device_delay_nanobind.cpp
)
