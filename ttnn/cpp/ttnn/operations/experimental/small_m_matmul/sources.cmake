# Source files for ttnn_op_experimental_small_m_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_SMALL_M_MATMUL_API_HEADERS
    small_m_matmul.hpp
    device/small_m_matmul_device_operation_types.hpp
    device/small_m_matmul_config.hpp
    device/small_m_matmul_plan.hpp
)

set(TTNN_OP_EXPERIMENTAL_SMALL_M_MATMUL_SRCS
    small_m_matmul.cpp
    device/small_m_matmul_config.cpp
    device/small_m_matmul_device_operation.cpp
    device/small_m_matmul_program_factory.cpp
)

# Nanobind registration sources, compiled into the shared `ttnn` Python module target (see CMakeLists.txt).
set(TTNN_OP_EXPERIMENTAL_SMALL_M_MATMUL_NANOBIND_SRCS small_m_matmul_nanobind.cpp)
