# Sources of moe_block/fused_tail. Owned by the fused_tail op: add its .cpp / API headers / nanobind sources here (absolute paths
# via CMAKE_CURRENT_LIST_DIR). Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/fused_tail.hpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS ${CMAKE_CURRENT_LIST_DIR}/fused_tail.cpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/fused_tail_nanobind.cpp)
