# Source files for ttnn_op_experimental_deepseek_prefill_masked_bincount.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MASKED_BINCOUNT_API_HEADERS masked_bincount.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MASKED_BINCOUNT_SRCS
    device/masked_bincount_device_operation.cpp
    device/masked_bincount_program_factory.cpp
    masked_bincount.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_MASKED_BINCOUNT_NANOBIND_SRCS masked_bincount_nanobind.cpp)
