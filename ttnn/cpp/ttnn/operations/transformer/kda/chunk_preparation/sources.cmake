# KDA chunk-preparation leaf sources.
set(TTNN_KDA_CHUNK_PREPARATION_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/kda_chunk_preparation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_chunk_preparation_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_chunk_preparation_program_factory.cpp
)
set(TTNN_KDA_CHUNK_PREPARATION_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/kda_chunk_preparation_nanobind.cpp)
