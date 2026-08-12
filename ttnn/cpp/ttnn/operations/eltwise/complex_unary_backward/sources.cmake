# Source files for ttnn_op_eltwise_complex_unary_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_COMPLEX_UNARY_BACKWARD_API_HEADERS complex_unary_backward.hpp)

set(TTNN_OP_ELTWISE_COMPLEX_UNARY_BACKWARD_SRCS
    complex_unary_backward.cpp
    device/complex_unary_backward_op.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/complex_unary_backward/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_COMPLEX_UNARY_BACKWARD_NANOBIND_SRCS complex_unary_backward_nanobind.cpp)
