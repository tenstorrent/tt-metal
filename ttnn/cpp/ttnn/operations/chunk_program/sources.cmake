# Source files for ttnn_op_chunk_program.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_CHUNK_PROGRAM_API_HEADERS chunk_program.hpp)

set(TTNN_OP_CHUNK_PROGRAM_SRCS
    chunk_program.cpp
    device/chunk_program_device_operation.cpp
    device/chunk_program_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/chunk_program/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_CHUNK_PROGRAM_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/chunk_program_nanobind.cpp)
