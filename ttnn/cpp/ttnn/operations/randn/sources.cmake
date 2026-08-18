# Source files for ttnn_op_randn.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_RANDN_SRCS
    randn.cpp
    device/randn_device_operation.cpp
    device/randn_program_factory.cpp
)

set(TTNN_OP_RANDN_API_HEADERS randn.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/randn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_RANDN_NANOBIND_SRCS randn_nanobind.cpp)
