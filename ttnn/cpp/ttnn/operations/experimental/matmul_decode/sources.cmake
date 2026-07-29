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

set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_NANOBIND_SRCS matmul_decode_nanobind.cpp)
