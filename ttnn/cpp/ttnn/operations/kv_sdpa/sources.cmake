# Source files for ttnn_op_kv_sdpa.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_KV_SDPA_SRCS
    kv_sdpa.cpp
    device/kv_sdpa_device_operation.cpp
    device/kv_sdpa_fused_program_factory.cpp
)

# Nanobind (Python binding) sources for this op. Wired onto the shared `ttnn`
# module target in CMakeLists.txt, per the new per-op registration convention.
set(TTNN_OP_KV_SDPA_NANOBIND_SRCS
    kv_sdpa_nanobind.cpp
)
