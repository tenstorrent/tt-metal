# Source files for ttnn_op_eltwise_ternary_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_TERNARY_BACKWARD_API_HEADERS ternary_backward.hpp)

set(TTNN_OP_ELTWISE_TERNARY_BACKWARD_SRCS ternary_backward.cpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/ternary_backward/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_TERNARY_BACKWARD_NANOBIND_SRCS ternary_backward_nanobind.cpp)
