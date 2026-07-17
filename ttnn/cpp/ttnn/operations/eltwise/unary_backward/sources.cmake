# Source files for ttnn_op_eltwise_unary_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_UNARY_BACKWARD_SRCS
    unary_backward.cpp
    tanh_bw/device/tanh_bw_device_operation.cpp
    tanh_bw/device/tanh_bw_program_factory.cpp
    gelu_bw/device/gelu_bw_device_operation.cpp
    gelu_bw/device/gelu_bw_program_factory.cpp
)

set(TTNN_OP_ELTWISE_UNARY_BACKWARD_API_HEADERS unary_backward.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/unary_backward/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_UNARY_BACKWARD_NANOBIND_SRCS unary_backward_nanobind.cpp)
