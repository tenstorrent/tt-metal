# Source files for ttnn_op_bringup_flat_routed_expert_ttnn.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_BRINGUP_FLAT_ROUTED_EXPERT_TTNN_API_HEADERS
    flat_routed_expert.hpp
    device/flat_routed_expert_plan.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/ttnn/bringup/flat_routed_expert_ttnn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_BRINGUP_FLAT_ROUTED_EXPERT_TTNN_NANOBIND_SRCS flat_routed_expert_nanobind.cpp)
