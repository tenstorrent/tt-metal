# Source files for ttnn_op_experimental_deepseek_prefill_routed_expert_combine_overlap.
# Module owners should update this file when adding/removing/renaming source files.

# What every op overlapping a routed expert with combine_fabric2d shares: combine's overlapped program factory
# (built in the routed expert's program, on rows 0-1), its collector, and the routed expert's hand-off to it.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ROUTED_EXPERT_COMBINE_OVERLAP_API_HEADERS
    device/combine_overlap_collector.hpp
    device/combine_overlap_program_factory.hpp
    device/routed_expert_handoff.hpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ROUTED_EXPERT_COMBINE_OVERLAP_SRCS
    device/combine_overlap_collector.cpp
    device/combine_overlap_program_factory.cpp
    device/routed_expert_handoff.cpp
)
