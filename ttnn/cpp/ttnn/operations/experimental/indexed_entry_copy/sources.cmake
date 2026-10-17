# Source files for ttnn_op_experimental_indexed_entry_copy.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_INDEXED_ENTRY_COPY_API_HEADERS indexed_entry_copy.hpp)

set(TTNN_OP_EXPERIMENTAL_INDEXED_ENTRY_COPY_SRCS
    device/indexed_entry_copy_device_operation.cpp
    device/indexed_entry_copy_program_factory.cpp
    indexed_entry_copy.cpp
)

# Registered on the shared `ttnn` Python module target from CMakeLists.txt (see its `if(TARGET ttnn)` block).
set(TTNN_OP_EXPERIMENTAL_INDEXED_ENTRY_COPY_NANOBIND_SRCS indexed_entry_copy_nanobind.cpp)
