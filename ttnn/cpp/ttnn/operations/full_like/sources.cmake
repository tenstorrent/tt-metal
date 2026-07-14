# Source files for ttnn_op_full_like.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_FULL_LIKE_SRCS full_like.cpp)

set(TTNN_OP_FULL_LIKE_API_HEADERS full_like.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/full_like/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_FULL_LIKE_NANOBIND_SRCS full_like_nanobind.cpp)
