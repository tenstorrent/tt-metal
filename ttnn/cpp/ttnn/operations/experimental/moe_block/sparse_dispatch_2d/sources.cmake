# Sources of moe_block/sparse_dispatch_2d. Kernels under device/kernels/ are picked up by ../CMakeLists.txt without
# listing.
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/sparse_dispatch_2d.hpp)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_MOE_BLOCK_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/sparse_dispatch_2d.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/sparse_dispatch_2d_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/sparse_dispatch_2d_program_factory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/sparse_dispatch_2d_plan.cpp
)
list(APPEND TTNN_OP_EXPERIMENTAL_MOE_BLOCK_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/sparse_dispatch_2d_nanobind.cpp)
