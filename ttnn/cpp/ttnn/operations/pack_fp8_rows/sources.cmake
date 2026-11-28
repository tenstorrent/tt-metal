# Source files for ttnn_op_pack_fp8_rows.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_PACK_FP8_ROWS_API_HEADERS pack_fp8_rows.hpp)

set(TTNN_OP_PACK_FP8_ROWS_SRCS
    pack_fp8_rows.cpp
    device/pack_fp8_rows_device_operation.cpp
    device/pack_fp8_rows_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/pack_fp8_rows/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_PACK_FP8_ROWS_NANOBIND_SRCS pack_fp8_rows_nanobind.cpp)
