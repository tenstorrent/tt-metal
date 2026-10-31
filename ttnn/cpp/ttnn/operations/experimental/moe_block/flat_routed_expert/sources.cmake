# Sources of moe_block/flat_routed_expert. Owned by the flat_routed_expert op: add its .cpp / API headers / nanobind sources here (absolute paths
# via CMAKE_CURRENT_LIST_DIR). Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/flat_routed_expert.hpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS ${CMAKE_CURRENT_LIST_DIR}/flat_routed_expert.cpp)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/flat_routed_expert_nanobind.cpp)
