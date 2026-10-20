# Source files for ttnn_op_experimental_deepseek_prefill_mhc_post_mix.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_POST_MIX_API_HEADERS mhc_post_mix.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_POST_MIX_SRCS
    device/mhc_post_mix_device_operation.cpp
    device/mhc_post_mix_program_factory.cpp
    mhc_post_mix.cpp
)

# Registered on the shared `ttnn` Python module target from this op's CMakeLists.txt (see the
# `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_POST_MIX_NANOBIND_SRCS mhc_post_mix_nanobind.cpp)
