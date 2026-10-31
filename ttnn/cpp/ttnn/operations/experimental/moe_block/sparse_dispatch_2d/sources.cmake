# Sources of moe_block/sparse_dispatch_2d. Owned by the sparse_dispatch_2d op: add its .cpp / API headers / nanobind sources here (absolute paths
# via CMAKE_CURRENT_LIST_DIR). Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/sparse_dispatch_2d.hpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS ${CMAKE_CURRENT_LIST_DIR}/sparse_dispatch_2d.cpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/sparse_dispatch_2d_nanobind.cpp)
