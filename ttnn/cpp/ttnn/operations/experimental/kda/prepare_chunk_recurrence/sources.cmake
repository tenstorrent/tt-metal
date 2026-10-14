set(TTNN_OP_EXPERIMENTAL_KDA_PREPARE_CHUNK_RECURRENCE_API_HEADERS prepare_chunk_recurrence.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_PREPARE_CHUNK_RECURRENCE_SRCS
    prepare_chunk_recurrence.cpp
    device/prepare_chunk_recurrence_device_operation.cpp
    device/prepare_chunk_recurrence_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_PREPARE_CHUNK_RECURRENCE_NANOBIND_SRCS
    prepare_chunk_recurrence_nanobind.cpp
    ../kda_nanobind.cpp
)
