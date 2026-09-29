set(TTNN_OP_EXPERIMENTAL_KDA_SIGMOID_GATED_RMS_NORM_API_HEADERS sigmoid_gated_rms_norm.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_SIGMOID_GATED_RMS_NORM_SRCS
    sigmoid_gated_rms_norm.cpp
    device/sigmoid_gated_rms_norm_device_operation.cpp
    device/sigmoid_gated_rms_norm_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_SIGMOID_GATED_RMS_NORM_NANOBIND_SRCS
    sigmoid_gated_rms_norm_nanobind.cpp
    ../kda_nanobind.cpp
)
