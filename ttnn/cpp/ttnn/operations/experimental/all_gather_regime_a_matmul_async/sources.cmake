# Source files for ttnn_op_experimental_all_gather_regime_a_matmul_async.

set(TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_API_HEADERS
    all_gather_regime_a_matmul_async.hpp
    device/all_gather_regime_a_matmul_async_plan.hpp
    device/all_gather_regime_a_matmul_async_device_operation_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_SRCS
    all_gather_regime_a_matmul_async.cpp
)

# Registered on the shared `ttnn` Python module target from this op's CMakeLists.txt (see the
# `if(TARGET ttnn)` block there). Listed here per the per-op nanobind convention.
set(TTNN_OP_EXPERIMENTAL_ALL_GATHER_REGIME_A_MATMUL_ASYNC_NANOBIND_SRCS
    all_gather_regime_a_matmul_async_nanobind.cpp
)
