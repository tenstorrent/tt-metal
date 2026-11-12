# Sources of moe_block/expert_slot_copy. Kernels under device/kernels/ are picked up by ../CMakeLists.txt without
# listing.
list(APPEND TTNN_OP_BRINGUP_MOE_BLOCK_MAIN_TTNN_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/expert_slot_copy.hpp)
list(
    APPEND
    TTNN_OP_BRINGUP_MOE_BLOCK_MAIN_TTNN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/expert_slot_copy.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/expert_slot_copy_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/expert_slot_copy_program_factory.cpp
)
list(APPEND TTNN_OP_BRINGUP_MOE_BLOCK_MAIN_TTNN_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/expert_slot_copy_nanobind.cpp)
