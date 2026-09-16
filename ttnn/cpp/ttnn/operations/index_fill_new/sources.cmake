# Source files for ttnn_op_index_fill_new.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_INDEX_FILL_NEW_API_HEADERS index_fill_new.hpp)

set(TTNN_OP_INDEX_FILL_NEW_SRCS
    device/index_fill_new_device_operation.cpp
    device/index_fill_new_program_factory.cpp
    index_fill_new.cpp
)

set(TTNN_OP_INDEX_FILL_NEW_NANOBIND_SRCS index_fill_new_nanobind.cpp)
