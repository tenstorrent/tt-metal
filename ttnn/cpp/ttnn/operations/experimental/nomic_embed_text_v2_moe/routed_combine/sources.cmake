# Source files for ttnn_op_experimental_nomic_embed_text_v2_moe_routed_combine.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTED_COMBINE_API_HEADERS routed_combine.hpp)

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTED_COMBINE_SRCS
    device/routed_combine_device_operation.cpp
    device/routed_combine_program_factory.cpp
    routed_combine.cpp
)

# Registered on the shared `ttnn` Python module target from this directory's CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTED_COMBINE_NANOBIND_SRCS routed_combine_nanobind.cpp)
