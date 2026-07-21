# Source files for ttnn_op_experimental_reshape.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_RESHAPE_API_HEADERS view.hpp)

set(TTNN_OP_EXPERIMENTAL_RESHAPE_SRCS view.cpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/reshape/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_RESHAPE_NANOBIND_SRCS view_nanobind.cpp)
