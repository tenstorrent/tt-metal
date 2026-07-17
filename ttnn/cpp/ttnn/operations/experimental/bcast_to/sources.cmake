# Source files for ttnn_op_experimental_bcast_to.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_BCAST_TO_API_HEADERS bcast_to.hpp)

set(TTNN_OP_EXPERIMENTAL_BCAST_TO_SRCS
    bcast_to.cpp
    device/bcast_to_device_operation.cpp
    device/bcast_to_program_factory.cpp
    device/bcast_to_utils.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/bcast_to/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_BCAST_TO_NANOBIND_SRCS bcast_to_nanobind.cpp)
