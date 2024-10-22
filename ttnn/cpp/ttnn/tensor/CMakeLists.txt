
set(TENSOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vector_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/size.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/alignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/page_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_layout.cpp

    CACHE INTERNAL "Tensor sources to reuse in ttnn build"
)
