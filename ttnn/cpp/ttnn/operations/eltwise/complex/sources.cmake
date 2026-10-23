# Source files for ttnn_op_eltwise_complex.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_COMPLEX_SRCS complex.cpp)

set(TTNN_OP_ELTWISE_COMPLEX_API_HEADERS complex.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/complex/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_COMPLEX_NANOBIND_SRCS complex_nanobind.cpp)
