set(TTNN_OP_EXPERIMENTAL_NVFP4_API_HEADERS dequantize_nvfp4.hpp)
set(TTNN_OP_EXPERIMENTAL_NVFP4_SRCS
    dequantize_nvfp4.cpp
    device/dequantize_nvfp4_device_operation.cpp
    device/dequantize_nvfp4_program_factory.cpp
)
set(TTNN_OP_EXPERIMENTAL_NVFP4_NANOBIND_SRCS dequantize_nvfp4_nanobind.cpp)
