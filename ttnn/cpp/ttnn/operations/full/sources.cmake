# Source files for ttnn_op_full.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_FULL_API_HEADERS full.hpp)

set(TTNN_OP_FULL_SRCS
    device/full_device_operation.cpp
    device/full_program_factory_interleaved.cpp
    device/full_program_factory_nd_sharded.cpp
    device/full_program_factory_sharded.cpp
    full.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/full/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_FULL_NANOBIND_SRCS full_nanobind.cpp)
