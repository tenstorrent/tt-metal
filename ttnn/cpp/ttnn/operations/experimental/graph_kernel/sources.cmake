# Source files for ttnn_op_experimental_graph_kernel.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_GRAPH_KERNEL_API_HEADERS graph_kernel.hpp)

set(TTNN_OP_EXPERIMENTAL_GRAPH_KERNEL_SRCS
    device/graph_kernel_device_operation.cpp
    device/graph_kernel_program_factory.cpp
    graph_kernel.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/graph_kernel/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_GRAPH_KERNEL_NANOBIND_SRCS graph_kernel_nanobind.cpp)
