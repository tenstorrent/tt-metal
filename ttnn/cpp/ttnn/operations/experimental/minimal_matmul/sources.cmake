# Source files for ttnn_op_experimental_minimal_matmul.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_MINIMAL_MATMUL_API_HEADERS
    minimal_matmul.hpp
    minimal_matmul_split.hpp
    device/minimal_matmul_device_operation_types.hpp
)

set(TTNN_OP_EXPERIMENTAL_MINIMAL_MATMUL_SRCS
    minimal_matmul.cpp
    minimal_matmul_split.cpp
    device/minimal_matmul_device_operation.cpp
    device/minimal_matmul_program_factory.cpp
    device/minimal_matmul_fabric_bound_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/minimal_matmul/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_MINIMAL_MATMUL_NANOBIND_SRCS
    minimal_matmul_nanobind.cpp
    minimal_matmul_split_nanobind.cpp
)
