# Source files for ttnn_op_experimental_unary_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_UNARY_BACKWARD_SRCS
    gelu_backward/device/gelu_backward_device_operation.cpp
    gelu_backward/device/gelu_backward_program_factory.cpp
    gelu_backward/gelu_backward.cpp
)

set(TTNN_OP_EXPERIMENTAL_UNARY_BACKWARD_API_HEADERS gelu_backward/gelu_backward.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/unary_backward/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_UNARY_BACKWARD_NANOBIND_SRCS gelu_backward/gelu_backward_nanobind.cpp)
