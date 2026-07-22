# Source files for ttnn_op_experimental_plusone.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_PLUSONE_API_HEADERS plusone.hpp)

set(TTNN_OP_EXPERIMENTAL_PLUSONE_SRCS
    device/plusone_device_operation.cpp
    device/plusone_program_factory.cpp
    plusone.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/plusone/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_PLUSONE_NANOBIND_SRCS plusone_nanobind.cpp)
