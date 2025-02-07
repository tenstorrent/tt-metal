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
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/tensor_types_to_flatbuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/tensor_types_from_flatbuffer.cpp
    CACHE INTERNAL
    "Tensor sources to reuse in ttnn build"
)

set(FBS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer/tensor_types.fbs)
get_filename_component(FBS_FILE_NAME ${FBS_FILE} NAME_WLE)
message(
    "CURRENT BIN DIR "
    ${CMAKE_CURRENT_BINARY_DIR}
)
set(FBS_GENERATED_HEADER "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers/${FBS_FILE_NAME}_generated.h")
add_custom_command(
    OUTPUT
        ${FBS_GENERATED_HEADER}
    COMMAND
        flatc --cpp --scoped-enums -o "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers/" ${FBS_FILE}
    DEPENDS
        flatc
        ${FBS_FILE}
    COMMENT "Generating FlatBuffers header ${FBS_GENERATED_HEADER}"
)
list(APPEND TENSOR_SRCS ${FBS_GENERATED_HEADER})
