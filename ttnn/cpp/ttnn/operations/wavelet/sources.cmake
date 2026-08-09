set(TTNN_OP_WAVELET_API_HEADERS
    wavelet.hpp
    wavelet_types.hpp
)

set(TTNN_OP_WAVELET_SRCS
    common/wavelet_host.cpp
    device/wavelet_1d_program_factory.cpp
    device/wavelet_2d_program_factory.cpp
    wavelet.cpp
)

set(TTNN_OP_WAVELET_NANOBIND_SRCS wavelet_nanobind.cpp)
