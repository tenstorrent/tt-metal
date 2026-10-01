# Source files for ttnn_op_eltwise_binary_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_BINARY_BACKWARD_SRCS binary_backward.cpp)

set(TTNN_OP_ELTWISE_BINARY_BACKWARD_API_HEADERS binary_backward.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/binary_backward/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_BINARY_BACKWARD_NANOBIND_SRCS binary_backward_nanobind.cpp)
