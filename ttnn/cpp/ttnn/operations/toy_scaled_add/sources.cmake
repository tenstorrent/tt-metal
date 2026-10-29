# Source files for ttnn_op_toy_scaled_add.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_TOY_SCALED_ADD_API_HEADERS toy_scaled_add.hpp)

set(TTNN_OP_TOY_SCALED_ADD_SRCS
    toy_scaled_add.cpp
    device/toy_scaled_add_device_operation.cpp
    device/toy_scaled_add_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/toy_scaled_add/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_TOY_SCALED_ADD_NANOBIND_SRCS toy_scaled_add_nanobind.cpp)
