# Source files for ttnn_op_bernoulli.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BERNOULLI_API_HEADERS bernoulli.hpp)

set(TTNN_OP_BERNOULLI_SRCS
    bernoulli.cpp
    device/bernoulli_device_operation.cpp
    device/bernoulli_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/bernoulli/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_BERNOULLI_NANOBIND_SRCS bernoulli_nanobind.cpp)
