
set(TENSOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_layout.cpp

    CACHE INTERNAL "Tensor sources to reuse in ttnn build"
)
