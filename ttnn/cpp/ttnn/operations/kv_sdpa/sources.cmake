# Source files for ttnn_op_kv_sdpa.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_KV_SDPA_SRCS
    kv_sdpa.cpp
    device/kv_sdpa_device_operation.cpp
    device/kv_sdpa_fused_program_factory.cpp
)

set(TTNN_OP_KV_SDPA_API_HEADERS kv_sdpa.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/kv_sdpa/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_KV_SDPA_NANOBIND_SRCS kv_sdpa_nanobind.cpp)
