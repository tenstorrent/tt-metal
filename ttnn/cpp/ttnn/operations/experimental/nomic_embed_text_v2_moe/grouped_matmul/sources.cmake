# Source files for ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_API_HEADERS grouped_matmul.hpp)

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_SRCS
    device/grouped_matmul_device_operation.cpp
    device/grouped_matmul_program_factory.cpp
    grouped_matmul.cpp
)

# Registered on the shared `ttnn` Python module target from this directory's CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_NANOBIND_SRCS grouped_matmul_nanobind.cpp)
