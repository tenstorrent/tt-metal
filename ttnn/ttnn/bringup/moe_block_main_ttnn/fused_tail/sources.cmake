# Sources of moe_block/fused_tail. Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(APPEND TTNN_OP_BRINGUP_MOE_BLOCK_MAIN_TTNN_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/fused_tail.hpp)
list(
    APPEND
    TTNN_OP_BRINGUP_MOE_BLOCK_MAIN_TTNN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/fused_tail.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/fused_tail_plan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/fused_tail_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/fused_tail_program_factory.cpp
)
list(APPEND TTNN_OP_BRINGUP_MOE_BLOCK_MAIN_TTNN_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/fused_tail_nanobind.cpp)
