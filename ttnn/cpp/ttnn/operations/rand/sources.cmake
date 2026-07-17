# Source files for ttnn_op_rand.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_RAND_SRCS
    rand.cpp
    device/rand_device_operation.cpp
    device/rand_program_factory.cpp
)

set(TTNN_OP_RAND_API_HEADERS rand.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/rand/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_RAND_NANOBIND_SRCS rand_nanobind.cpp)
