# Source files for ttnn_op_loss.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_LOSS_API_HEADERS
    loss.hpp
    loss_types.hpp
)

set(TTNN_OP_LOSS_SRCS loss.cpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/loss/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_LOSS_NANOBIND_SRCS loss_nanobind.cpp)
