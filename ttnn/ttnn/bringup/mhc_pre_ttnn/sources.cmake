# Source files for ttnn_op_bringup_mhc_pre_ttnn (the C++ host side of ttnn.bringup.mhc_pre).
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BRINGUP_MHC_PRE_TTNN_API_HEADERS
    mhc_pre_ttnn.hpp
    mhc_pre_xing.hpp
)

set(TTNN_OP_BRINGUP_MHC_PRE_TTNN_SRCS
    device/mhc_pre_ttnn_device_operation.cpp
    device/mhc_pre_ttnn_program_factory.cpp
    mhc_pre_ttnn.cpp
    device/mhc_pre_xing_device_operation.cpp
    device/mhc_pre_xing_program_factory.cpp
    mhc_pre_xing.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/ttnn/bringup/mhc_pre_ttnn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_BRINGUP_MHC_PRE_TTNN_NANOBIND_SRCS mhc_pre_ttnn_nanobind.cpp)
