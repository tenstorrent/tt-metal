# Sources of moe_block/routed_experts_combine (absolute paths via CMAKE_CURRENT_LIST_DIR). Kernels under device/kernels/
# are picked up by ../CMakeLists.txt without listing.
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/routed_experts_combine.hpp)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/routed_experts_combine.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/combine_host.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/combine_program.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/routed_experts_combine_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/routed_experts_combine_program_factory.cpp
)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/routed_experts_combine_nanobind.cpp)
