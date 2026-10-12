# Source files for ttnn_op_experimental_deepseek_prefill_shared_expert_ffn.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_SHARED_EXPERT_FFN_API_HEADERS
    shared_expert_ffn.hpp
    device/shared_expert_ffn_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_SHARED_EXPERT_FFN_SRCS
    device/shared_expert_ffn_device_operation.cpp
    shared_expert_ffn.cpp
)

# Registered on the shared `ttnn` Python module target from this op's CMakeLists.txt (see the
# `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_SHARED_EXPERT_FFN_NANOBIND_SRCS shared_expert_ffn_nanobind.cpp)
