set(TTNN_OP_EXPERIMENTAL_KDA_GDN_DECODE_STEP_API_HEADERS gdn_decode_step.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_GDN_DECODE_STEP_SRCS
    gdn_decode_step.cpp
    device/gdn_decode_step_device_operation.cpp
    device/gdn_decode_step_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_GDN_DECODE_STEP_NANOBIND_SRCS
    gdn_decode_step_nanobind.cpp
    ../kda_nanobind.cpp
)
