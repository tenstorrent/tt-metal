# Source files for ttnn_op_experimental_deepseek_prefill_mhc_split_sinkhorn.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_SPLIT_SINKHORN_API_HEADERS mhc_split_sinkhorn.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_SPLIT_SINKHORN_SRCS
    device/mhc_split_sinkhorn_device_operation.cpp
    device/mhc_split_sinkhorn_program_factory.cpp
    mhc_split_sinkhorn.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MHC_SPLIT_SINKHORN_NANOBIND_SRCS mhc_split_sinkhorn_nanobind.cpp)
