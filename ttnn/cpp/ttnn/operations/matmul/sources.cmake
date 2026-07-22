# Source files for ttnn_op_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_MATMUL_SRCS
    matmul.cpp
    device/config/matmul_program_config.cpp
    device/config/matmul_auto_tuner.cpp
    device/matmul_device_operation.cpp
    device/utilities/matmul_utilities.cpp
    device/factory/matmul_multicore_program_factory.cpp
    device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp
    device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp
    device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp
    device/factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp
    device/factory/matmul_multicore_reuse_optimized_program_factory.cpp
    device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp
    device/sparse/sparse_matmul_device_operation.cpp
)

set(TTNN_OP_MATMUL_API_HEADERS
    matmul.hpp
    device/matmul_device_operation.hpp
    device/matmul_device_operation_types.hpp
    device/config/matmul_program_config.hpp
    device/config/matmul_program_config_types.hpp
    device/config/matmul_auto_tuner.hpp
    shared_with_host/activation_type.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_MATMUL_NANOBIND_SRCS matmul_nanobind.cpp)
