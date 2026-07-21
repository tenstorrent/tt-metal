# Source files for ttnn_op_conv.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_CONV_SRCS conv_placeholder.cpp)

set(TTNN_OP_CONV_API_HEADERS)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/conv/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_CONV_NANOBIND_SRCS conv_nanobind.cpp)
