# Source files for ttnn_op_experimental_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_MATMUL_API_HEADERS
    attn_matmul/attn_matmul.hpp
    group_attn_matmul/group_attn_matmul.hpp
)

set(TTNN_OP_EXPERIMENTAL_MATMUL_SRCS
    attn_matmul/attn_matmul.cpp
    attn_matmul/device/attn_matmul_device_operation.cpp
    attn_matmul/device/attn_matmul_program_factory.cpp
    group_attn_matmul/device/group_attn_matmul_device_operation.cpp
    group_attn_matmul/device/group_attn_matmul_program_factory.cpp
    group_attn_matmul/group_attn_matmul.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_MATMUL_NANOBIND_SRCS
    attn_matmul/attn_matmul_nanobind.cpp
    group_attn_matmul/group_attn_matmul_nanobind.cpp
)
