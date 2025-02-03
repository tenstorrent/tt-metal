set(TENSOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_spec.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shape/shape_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shape/shape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layout/alignment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layout/page_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layout/tensor_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/xtensor/partition.cpp
    CACHE INTERNAL
    "Tensor sources to reuse in ttnn build"
)
