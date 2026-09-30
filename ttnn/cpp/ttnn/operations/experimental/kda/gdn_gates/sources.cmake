set(TTNN_OP_EXPERIMENTAL_KDA_GDN_GATES_API_HEADERS gdn_gates.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_GDN_GATES_SRCS
    gdn_gates.cpp
    device/gdn_gates_device_operation.cpp
    device/gdn_gates_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_GDN_GATES_NANOBIND_SRCS
    gdn_gates_nanobind.cpp
    ../kda_nanobind.cpp
)
