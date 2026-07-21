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

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/regime_a_matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_REGIME_A_MATMUL_NANOBIND_SRCS
    regime_a_matmul_nanobind.cpp
)
