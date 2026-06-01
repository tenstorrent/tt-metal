# Source files for ttnn_op_experimental_deltanet.

set(TTNN_OP_EXPERIMENTAL_DELTANET_SRCS
    device/deltanet_decode_device_operation.cpp
    device/deltanet_decode_program_factory.cpp
    deltanet_decode.cpp
    device/deltanet_full_device_operation.cpp
    device/deltanet_full_program_factory.cpp
    deltanet_decode_full.cpp
    device/deltanet_prefill_device_operation.cpp
    device/deltanet_prefill_program_factory.cpp
    deltanet_prefill_full.cpp
)

set(TTNN_OP_EXPERIMENTAL_DELTANET_API_HEADERS
    deltanet_decode.hpp
    deltanet_decode_full.hpp
    deltanet_prefill_full.hpp
)
