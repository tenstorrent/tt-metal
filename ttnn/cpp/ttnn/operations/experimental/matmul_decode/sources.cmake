# Source files for ttnn_op_experimental_matmul_decode.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_API_HEADERS
    matmul_decode.hpp
    device/matmul_decode_device_operation.hpp
)

set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_SRCS
    matmul_decode.cpp
    device/matmul_decode_device_operation.cpp
    device/full_width_sharded_program_factory.cpp
    device/partial_width_sharded_program_factory.cpp
    device/batched_width_sharded_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/matmul_decode/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_NANOBIND_SRCS matmul_decode_nanobind.cpp)
