# Source files for ttnn_op_experimental_nomic_embed_text_v2_moe_grouped_matmul_w2.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_W2_API_HEADERS grouped_matmul_w2.hpp)

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_W2_SRCS
    device/grouped_matmul_w2_device_operation.cpp
    device/grouped_matmul_w2_program_factory.cpp
    grouped_matmul_w2.cpp
)

# Registered on the shared `ttnn` Python module target from this directory's CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_GROUPED_MATMUL_W2_NANOBIND_SRCS grouped_matmul_w2_nanobind.cpp)
