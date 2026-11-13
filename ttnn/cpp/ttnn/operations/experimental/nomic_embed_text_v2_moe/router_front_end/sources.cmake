# Source files for ttnn_op_experimental_nomic_embed_text_v2_moe_router_front_end.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTER_FRONT_END_API_HEADERS router_front_end.hpp)

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTER_FRONT_END_SRCS
    device/router_front_end_device_operation.cpp
    device/router_front_end_program_factory.cpp
    router_front_end.cpp
)

# Registered on the shared `ttnn` Python module target from this directory's CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTER_FRONT_END_NANOBIND_SRCS router_front_end_nanobind.cpp)
