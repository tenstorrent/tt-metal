# Source files for ttnn_op_index_fill.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_INDEX_FILL_API_HEADERS index_fill.hpp)

set(TTNN_OP_INDEX_FILL_SRCS
    device/index_fill_device_operation.cpp
    device/index_fill_multi_core_factory.cpp
    index_fill.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/index_fill/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_INDEX_FILL_NANOBIND_SRCS index_fill_nanobind.cpp)
