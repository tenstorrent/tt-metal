# Source files for ttnn_op_creation.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_CREATION_SRCS creation.cpp)

set(TTNN_OP_CREATION_API_HEADERS creation.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/creation/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_CREATION_NANOBIND_SRCS
    creation_nanobind.cpp
    creation.hpp
)
