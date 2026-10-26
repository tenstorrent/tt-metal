# Source files for ttnn_op_bringup_prepared_program.

set(TTNN_OP_BRINGUP_PREPARED_PROGRAM_API_HEADERS prepared_program.hpp)

set(TTNN_OP_BRINGUP_PREPARED_PROGRAM_SRCS
    device/prepared_program_device_operation.cpp
    prepared_program.cpp
)

# Registered on the shared `ttnn` Python module target from CMakeLists.txt (the `if(TARGET ttnn)` block there).
set(TTNN_OP_BRINGUP_PREPARED_PROGRAM_NANOBIND_SRCS prepared_program_nanobind.cpp)
