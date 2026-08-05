# KDA causal-convolution operation registration.
set(TTNN_KDA_CAUSAL_CONV_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/kda_causal_conv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_causal_conv_device_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/device/kda_causal_conv_program_factory.cpp
)

set(TTNN_KDA_CAUSAL_CONV_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/kda_causal_conv_nanobind.cpp)
