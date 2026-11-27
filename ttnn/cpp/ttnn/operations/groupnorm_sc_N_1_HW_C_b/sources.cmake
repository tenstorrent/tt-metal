# Source files for ttnn_op_groupnorm_sc_N_1_HW_C_b.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_GROUPNORM_SC_N_1_HW_C_B_API_HEADERS groupnorm_sc_N_1_HW_C_b.hpp)

set(TTNN_OP_GROUPNORM_SC_N_1_HW_C_B_SRCS
    groupnorm_sc_N_1_HW_C_b.cpp
    device/groupnorm_sc_N_1_HW_C_b_device_operation.cpp
    device/groupnorm_sc_N_1_HW_C_b_program_factory.cpp
    device/groupnorm_sc_N_1_HW_C_b_planner.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/groupnorm_sc_N_1_HW_C_b/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_GROUPNORM_SC_N_1_HW_C_B_NANOBIND_SRCS groupnorm_sc_N_1_HW_C_b_nanobind.cpp)
