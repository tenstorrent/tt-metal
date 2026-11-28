# Source files for ttnn_op_ckv_bfp4_rows.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_CKV_BFP4_ROWS_API_HEADERS ckv_bfp4_rows.hpp)

set(TTNN_OP_CKV_BFP4_ROWS_SRCS
    ckv_bfp4_rows.cpp
    device/ckv_bfp4_rows_device_operation.cpp
    device/ckv_bfp4_rows_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/ckv_bfp4_rows/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_CKV_BFP4_ROWS_NANOBIND_SRCS ckv_bfp4_rows_nanobind.cpp)
