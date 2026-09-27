# Source files for ttnn_op_bringup_offset_cumsum.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BRINGUP_OFFSET_CUMSUM_API_HEADERS offset_cumsum.hpp)

set(TTNN_OP_BRINGUP_OFFSET_CUMSUM_SRCS
    device/offset_cumsum_device_operation.cpp
    device/offset_cumsum_program_factory.cpp
    offset_cumsum.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/ttnn/bringup/offset_cumsum/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_BRINGUP_OFFSET_CUMSUM_NANOBIND_SRCS offset_cumsum_nanobind.cpp)
