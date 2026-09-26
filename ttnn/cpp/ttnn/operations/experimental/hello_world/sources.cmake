# Source files for ttnn_op_experimental_hello_world.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_HELLO_WORLD_SRCS
    device/hello_world_device_operation.cpp
    device/hello_world_program_factory.cpp
    hello_world.cpp
)

set(TTNN_OP_EXPERIMENTAL_HELLO_WORLD_API_HEADERS hello_world.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/hello_world/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_HELLO_WORLD_NANOBIND_SRCS hello_world_nanobind.cpp)
