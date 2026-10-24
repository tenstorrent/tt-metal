# Source files for ttnn_op_bringup_sdpa (written by fork_op.py: the source op had no target of its own).

set(TTNN_OP_BRINGUP_SDPA_API_HEADERS
    sdpa.hpp
    sparse_sdpa.hpp
    sparse_sdpa_msa.hpp
)

set(TTNN_OP_BRINGUP_SDPA_SRCS
    device/exp_ring_joint_sdpa_device_operation.cpp
    device/exp_ring_joint_sdpa_program_factory.cpp
    device/joint_sdpa_device_operation.cpp
    device/joint_sdpa_program_factory.cpp
    device/ring_distributed_sdpa_device_operation.cpp
    device/ring_distributed_sdpa_program_factory.cpp
    device/ring_fusion.cpp
    device/ring_joint_sdpa_device_operation.cpp
    device/ring_joint_sdpa_program_factory.cpp
    device/sdpa_device_operation.cpp
    device/sdpa_perf_model.cpp
    device/sdpa_program_factory.cpp
    device/sliding_halo_layout.cpp
    device/sparse_sdpa_device_operation.cpp
    device/sparse_sdpa_msa_device_operation.cpp
    device/sparse_sdpa_msa_program_factory.cpp
    device/sparse_sdpa_program_factory.cpp
    sdpa.cpp
    sparse_sdpa.cpp
    sparse_sdpa_msa.cpp
)

set(TTNN_OP_BRINGUP_SDPA_NANOBIND_SRCS sdpa_nanobind.cpp)
