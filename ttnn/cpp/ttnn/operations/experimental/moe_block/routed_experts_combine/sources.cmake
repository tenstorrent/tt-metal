# Sources of moe_block/routed_experts_combine. Owned by the routed_experts_combine op: add its .cpp / API headers / nanobind sources here (absolute paths
# via CMAKE_CURRENT_LIST_DIR). Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/routed_experts_combine.hpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS ${CMAKE_CURRENT_LIST_DIR}/routed_experts_combine.cpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/routed_experts_combine_nanobind.cpp)
