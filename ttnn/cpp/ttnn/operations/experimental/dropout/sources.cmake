# Source files for ttnn_op_experimental_dropout.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DROPOUT_SRCS
    device/dropout_device_operation.cpp
    device/dropout_program_factory.cpp
    dropout.cpp
)

set(TTNN_OP_EXPERIMENTAL_DROPOUT_API_HEADERS dropout.hpp)
