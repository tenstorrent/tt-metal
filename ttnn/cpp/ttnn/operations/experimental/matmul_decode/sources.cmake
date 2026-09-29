# Source files for ttnn_op_experimental_matmul_decode.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_API_HEADERS
    matmul_decode.hpp
    packed_weight_spec.hpp
    device/matmul_decode_device_operation.hpp
    device/matmul_decode_descriptor.hpp
)

set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_SRCS
    matmul_decode.cpp
    device/matmul_decode_device_operation.cpp
    device/full_width_sharded_program_factory.cpp
    device/partial_width_sharded_program_factory.cpp
    device/batched_width_sharded_program_factory.cpp
    device/matmul_decode_descriptor.cpp
)

# Device kernels installed with the op (FILE_SET kernels). Listed here rather than
# globbed in CMakeLists.txt so add/remove/rename does not touch infra-owned build logic.
set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_KERNELS
    device/kernels/compute/compute_batched_width_sharded.cpp
    device/kernels/compute/compute_full_width_ring_gather.cpp
    device/kernels/compute/compute_full_width_sharded.cpp
    device/kernels/compute/compute_partial_width_sharded.cpp
    device/kernels/dataflow/all_gather_local_output.hpp
    device/kernels/dataflow/full_width_rms_norm_transport.hpp
    device/kernels/dataflow/reader_batched_width_sharded.cpp
    device/kernels/dataflow/reader_full_width_ring_gather.cpp
    device/kernels/dataflow/reader_full_width_sharded.cpp
    device/kernels/dataflow/reader_partial_width_ring_gather.cpp
    device/kernels/dataflow/reader_partial_width_sharded.cpp
    device/kernels/dataflow/writer_batched_width_sharded.cpp
    device/kernels/dataflow/writer_full_width_all_gather.cpp
    device/kernels/dataflow/writer_full_width_output_mcast.cpp
    device/kernels/dataflow/writer_full_width_rms_norm.cpp
    device/kernels/dataflow/writer_partial_width_sharded.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/matmul_decode/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_MATMUL_DECODE_NANOBIND_SRCS matmul_decode_nanobind.cpp)
