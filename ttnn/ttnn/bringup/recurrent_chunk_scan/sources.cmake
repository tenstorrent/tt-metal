set(TTNN_OP_BRINGUP_RECURRENT_CHUNK_SCAN_API_HEADERS recurrent_chunk_scan.hpp)

set(TTNN_OP_BRINGUP_RECURRENT_CHUNK_SCAN_SRCS
    recurrent_chunk_scan.cpp
    device/recurrent_chunk_scan_device_operation.cpp
    device/recurrent_chunk_scan_program_factory.cpp
)

set(TTNN_OP_BRINGUP_RECURRENT_CHUNK_SCAN_NANOBIND_SRCS recurrent_chunk_scan_nanobind.cpp)
