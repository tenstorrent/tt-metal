# Sources of moe_block/common (shared by every op; included by ../CMakeLists.txt). Changes here are the lead's.
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/moe_block_config.hpp
    ${CMAKE_CURRENT_LIST_DIR}/mesh_routing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/port_placement.hpp
    ${CMAKE_CURRENT_LIST_DIR}/moe_block_resources.hpp
    ${CMAKE_CURRENT_LIST_DIR}/l1_budget.hpp
    ${CMAKE_CURRENT_LIST_DIR}/address_patches.hpp
    ${CMAKE_CURRENT_LIST_DIR}/chip_program.hpp
    ${CMAKE_CURRENT_LIST_DIR}/args_table.hpp
    ${CMAKE_CURRENT_LIST_DIR}/host_floats.hpp
)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/moe_block_config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mesh_routing.cpp
    ${CMAKE_CURRENT_LIST_DIR}/port_placement.cpp
    ${CMAKE_CURRENT_LIST_DIR}/moe_block_resources.cpp
    ${CMAKE_CURRENT_LIST_DIR}/l1_budget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/address_patches.cpp
    ${CMAKE_CURRENT_LIST_DIR}/chip_program.cpp
    ${CMAKE_CURRENT_LIST_DIR}/args_table.cpp
)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/common_nanobind.cpp)
