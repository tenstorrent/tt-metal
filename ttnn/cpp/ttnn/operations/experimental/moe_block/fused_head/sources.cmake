# Sources of moe_block/fused_head. Owned by the fused_head op: add its .cpp / API headers / nanobind sources here (absolute paths
# via CMAKE_CURRENT_LIST_DIR). Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/fused_head.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_sizing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_plan.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_program.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/fused_head_device_operation_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/fused_head_device_operation.hpp
)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/fused_head.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_sizing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_plan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_program.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/fused_head_device_operation.cpp
)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/fused_head_nanobind.cpp)
