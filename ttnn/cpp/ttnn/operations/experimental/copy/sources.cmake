# Source files for ttnn_op_experimental_copy.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_COPY_API_HEADERS typecast/typecast.hpp)

set(TTNN_OP_EXPERIMENTAL_COPY_SRCS typecast/typecast.cpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/copy/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_COPY_NANOBIND_SRCS typecast/typecast_nanobind.cpp)
