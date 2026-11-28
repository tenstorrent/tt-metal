# Source files for ttnn_op_fabric_reduce_scatter.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_FABRIC_REDUCE_SCATTER_API_HEADERS fabric_reduce_scatter.hpp)

set(TTNN_OP_FABRIC_REDUCE_SCATTER_SRCS
    fabric_reduce_scatter.cpp
    device/fabric_reduce_scatter_device_operation.cpp
    device/fabric_reduce_scatter_program_factory.cpp
    device/fabric_reduce_scatter_planner.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/fabric_reduce_scatter/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_FABRIC_REDUCE_SCATTER_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/fabric_reduce_scatter_nanobind.cpp)
