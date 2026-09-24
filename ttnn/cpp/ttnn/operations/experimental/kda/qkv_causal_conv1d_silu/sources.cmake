set(TTNN_OP_EXPERIMENTAL_KDA_QKV_CAUSAL_CONV1D_SILU_API_HEADERS qkv_causal_conv1d_silu.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_QKV_CAUSAL_CONV1D_SILU_SRCS
    qkv_causal_conv1d_silu.cpp
    device/qkv_causal_conv1d_silu_device_operation.cpp
    device/qkv_causal_conv1d_silu_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_QKV_CAUSAL_CONV1D_SILU_NANOBIND_SRCS
    qkv_causal_conv1d_silu_nanobind.cpp
    ../kda_nanobind.cpp
)
