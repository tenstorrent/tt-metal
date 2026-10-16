# Source files for ttnn_op_experimental_nomic_embed_text_v2_moe_routing_metadata.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTING_METADATA_API_HEADERS routing_metadata.hpp)

set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTING_METADATA_SRCS
    device/routing_metadata_device_operation.cpp
    device/routing_metadata_program_factory.cpp
    routing_metadata.cpp
)

# Registered on the shared `ttnn` Python module target from this directory's CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_NOMIC_EMBED_TEXT_V2_MOE_ROUTING_METADATA_NANOBIND_SRCS routing_metadata_nanobind.cpp)
