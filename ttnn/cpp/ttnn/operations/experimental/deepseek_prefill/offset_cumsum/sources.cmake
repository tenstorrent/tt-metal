# Source files for ttnn_op_experimental_deepseek_prefill_offset_cumsum.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_OFFSET_CUMSUM_API_HEADERS offset_cumsum.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_OFFSET_CUMSUM_SRCS
    device/offset_cumsum_device_operation.cpp
    device/offset_cumsum_program_factory.cpp
    offset_cumsum.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_OFFSET_CUMSUM_NANOBIND_SRCS offset_cumsum_nanobind.cpp)
