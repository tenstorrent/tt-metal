# Source files for ttnn_op_experimental_regime_a_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_REGIME_A_MATMUL_API_HEADERS
    regime_a_matmul.hpp
    device/regime_a_matmul_device_operation_types.hpp
    device/regime_a_matmul_config.hpp
    device/regime_a_matmul_plan.hpp
)

set(TTNN_OP_EXPERIMENTAL_REGIME_A_MATMUL_SRCS
    regime_a_matmul.cpp
    device/regime_a_matmul_config.cpp
    device/regime_a_matmul_device_operation.cpp
    device/regime_a_matmul_program_factory.cpp
)
