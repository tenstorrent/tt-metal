# Sources of moe_block/flat_routed_expert. Kernels under device/kernels/ are picked up by ../CMakeLists.txt.
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/flat_routed_expert.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_geometry.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_program.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_device_operation.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_device_operation_types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_plan.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_program_factory.hpp
)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/flat_routed_expert.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_geometry.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_options.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_plan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/flat_routed_expert_program_factory.cpp
)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/flat_routed_expert_nanobind.cpp)
