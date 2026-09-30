# Source files for ttnn_op_bringup_mhc_post_ttnn (the C++ host side of ttnn.bringup.mhc_post).
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BRINGUP_MHC_POST_TTNN_API_HEADERS mhc_post_ttnn.hpp)

set(TTNN_OP_BRINGUP_MHC_POST_TTNN_SRCS
    device/mhc_post_ttnn_device_operation.cpp
    device/mhc_post_ttnn_program_factory.cpp
    mhc_post_ttnn.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/ttnn/bringup/mhc_post_ttnn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_BRINGUP_MHC_POST_TTNN_NANOBIND_SRCS mhc_post_ttnn_nanobind.cpp)
