# Source files for ttnn_op_experimental_modulated_deform_im2col.

set(TTNN_OP_EXPERIMENTAL_MODULATED_DEFORM_IM2COL_API_HEADERS modulated_deform_im2col.hpp)

set(TTNN_OP_EXPERIMENTAL_MODULATED_DEFORM_IM2COL_SRCS
    modulated_deform_im2col.cpp
    device/modulated_deform_im2col_device_operation.cpp
    device/modulated_deform_im2col_program_factory.cpp
    device/modulated_deform_conv_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_MODULATED_DEFORM_IM2COL_NANOBIND_SRCS modulated_deform_im2col_nanobind.cpp)
