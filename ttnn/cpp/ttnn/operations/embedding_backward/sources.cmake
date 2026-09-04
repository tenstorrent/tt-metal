# Source files for ttnn_op_embedding_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EMBEDDING_BACKWARD_SRCS
    embedding_backward.cpp
    device/embedding_backward_device_operation.cpp
    device/embedding_backward_program_descriptor.cpp
)

set(TTNN_OP_EMBEDDING_BACKWARD_API_HEADERS embedding_backward.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/embedding_backward/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EMBEDDING_BACKWARD_NANOBIND_SRCS embedding_backward_nanobind.cpp)
