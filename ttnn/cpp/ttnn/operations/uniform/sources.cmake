# Source files for ttnn_op_uniform.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_UNIFORM_API_HEADERS uniform.hpp)

set(TTNN_OP_UNIFORM_SRCS
    device/uniform_device_operation.cpp
    device/uniform_program_factory.cpp
    uniform.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/uniform/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_UNIFORM_NANOBIND_SRCS uniform_nanobind.cpp)
