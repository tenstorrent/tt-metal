# Sources of moe_block/moe_tail. Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/moe_tail.hpp)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/moe_tail.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/moe_tail_plan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/moe_tail_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/moe_tail_program_factory.cpp
)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/moe_tail_nanobind.cpp)
