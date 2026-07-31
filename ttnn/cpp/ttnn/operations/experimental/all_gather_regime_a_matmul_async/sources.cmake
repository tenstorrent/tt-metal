# Source files for ttnn_op_experimental_all_gather_regime_a_matmul_async.
# Module owners should update this file when adding/removing/renaming source files.
#
# NOTE: the config (RegimeAMatmulConfig + auto_select_config picker) and the host planner
# (regime_a_matmul_plan.hpp) are REUSED from ttnn_op_experimental_regime_a_matmul rather than
# duplicated here — see CMakeLists.txt for the link dependency.

set(TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_API_HEADERS
    all_gather_regime_a_matmul_async.hpp
    device/all_gather_regime_a_matmul_async_device_operation_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_SRCS
    all_gather_regime_a_matmul_async.cpp
    device/all_gather_regime_a_matmul_async_device_operation.cpp
    device/all_gather_regime_a_matmul_async_program_factory.cpp
)

# Nanobind registration sources, compiled into the shared `ttnn` Python module target (see CMakeLists.txt).
set(TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_NANOBIND_SRCS all_gather_regime_a_matmul_async_nanobind.cpp)
