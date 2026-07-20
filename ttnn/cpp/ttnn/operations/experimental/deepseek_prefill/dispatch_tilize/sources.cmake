# Source files for ttnn_op_experimental_deepseek_prefill_dispatch_tilize.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_TILIZE_API_HEADERS dispatch_tilize.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_TILIZE_SRCS
    device/dispatch_tilize_device_operation.cpp
    device/dispatch_tilize_program_factory.cpp
    dispatch_tilize.cpp
)

# Registered on the shared `ttnn` Python module target from CMakeLists.txt (the `if(TARGET ttnn)` block).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_DISPATCH_TILIZE_NANOBIND_SRCS dispatch_tilize_nanobind.cpp)
