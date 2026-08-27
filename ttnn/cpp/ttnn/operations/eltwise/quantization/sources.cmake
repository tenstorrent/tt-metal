# Source files for ttnn_op_eltwise_quantization.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_QUANTIZATION_SRCS quantization.cpp)

set(TTNN_OP_ELTWISE_QUANTIZATION_API_HEADERS quantization.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/quantization/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_QUANTIZATION_NANOBIND_SRCS quantization_nanobind.cpp)
