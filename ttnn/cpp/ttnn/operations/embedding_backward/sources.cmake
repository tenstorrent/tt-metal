# Source files for ttnn_op_embedding_backward.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EMBEDDING_BACKWARD_SRCS
    embedding_backward.cpp
    device/embedding_backward_device_operation.cpp
    device/embedding_backward_program_factory.cpp
)

set(TTNN_OP_EMBEDDING_BACKWARD_API_HEADERS embedding_backward.hpp)
