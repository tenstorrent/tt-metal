# Source files for ttnn_op_experimental_subdevice_sync.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_SUBDEVICE_SYNC_API_HEADERS subdevice_sync.hpp)

set(TTNN_OP_EXPERIMENTAL_SUBDEVICE_SYNC_SRCS
    device/subdevice_sync_device_operation.cpp
    subdevice_sync.cpp
)

set(TTNN_OP_EXPERIMENTAL_SUBDEVICE_SYNC_NANOBIND_SRCS subdevice_sync_nanobind.cpp)
