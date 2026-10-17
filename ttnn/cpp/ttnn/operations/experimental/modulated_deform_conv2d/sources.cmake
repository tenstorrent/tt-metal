# Source files for ttnn_op_experimental_modulated_deform_conv2d.

set(TTNN_OP_EXPERIMENTAL_MODULATED_DEFORM_CONV2D_API_HEADERS modulated_deform_conv2d.hpp)

set(TTNN_OP_EXPERIMENTAL_MODULATED_DEFORM_CONV2D_SRCS
    modulated_deform_conv2d.cpp
    device/modulated_deform_conv2d_device_operation.cpp
    device/modulated_deform_conv2d_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_MODULATED_DEFORM_CONV2D_NANOBIND_SRCS modulated_deform_conv2d_nanobind.cpp)
