# Source files for ttnn_op_experimental_deepseek_prefill_moe_fanout_reach.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_FANOUT_REACH_API_HEADERS moe_fanout_reach.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_FANOUT_REACH_SRCS
    device/moe_fanout_reach_device_operation.cpp
    device/moe_fanout_reach_program_factory.cpp
    moe_fanout_reach.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fanout_reach/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MOE_FANOUT_REACH_NANOBIND_SRCS moe_fanout_reach_nanobind.cpp)
