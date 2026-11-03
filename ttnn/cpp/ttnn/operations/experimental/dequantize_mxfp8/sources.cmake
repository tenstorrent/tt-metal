set(TTNN_OP_EXPERIMENTAL_MXFP8_API_HEADERS dequantize_mxfp8.hpp)
set(TTNN_OP_EXPERIMENTAL_MXFP8_SRCS
    dequantize_mxfp8.cpp
    device/dequantize_mxfp8_device_operation.cpp
    device/dequantize_mxfp8_program_factory.cpp
)
set(TTNN_OP_EXPERIMENTAL_MXFP8_NANOBIND_SRCS dequantize_mxfp8_nanobind.cpp)
