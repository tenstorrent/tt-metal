# Sources of moe_block/moe_head. Owned by the moe_head op: add its .cpp / API headers / nanobind sources here (absolute paths
# via CMAKE_CURRENT_LIST_DIR). Kernels under device/kernels/ are picked up by ../CMakeLists.txt without listing.
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/moe_head.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_sizing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_plan.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_program.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/moe_head_device_operation_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/moe_head_device_operation.hpp
)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/moe_head.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_sizing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_engine_program.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_gather_program.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_plan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_program.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_router_program.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_row_blocks.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_weights.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/head_workload.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/moe_head_device_operation.cpp
)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/moe_head_nanobind.cpp)
