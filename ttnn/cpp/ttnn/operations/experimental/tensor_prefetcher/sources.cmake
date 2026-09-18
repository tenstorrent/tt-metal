# Source files for ttnn_op_experimental_tensor_prefetcher.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_TENSOR_PREFETCHER_SRCS tensor_prefetcher.cpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/tensor_prefetcher/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_TENSOR_PREFETCHER_NANOBIND_SRCS tensor_prefetcher_nanobind.cpp)
