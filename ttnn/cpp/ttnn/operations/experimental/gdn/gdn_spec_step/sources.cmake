set(TTNN_OP_EXPERIMENTAL_GDN_SPEC_STEP_API_HEADERS gdn_spec_step.hpp)

set(TTNN_OP_EXPERIMENTAL_GDN_SPEC_STEP_SRCS
    gdn_spec_step.cpp
    device/gdn_spec_step_device_operation.cpp
    device/gdn_spec_step_program_factory.cpp
)

# ../gdn_nanobind.cpp is already compiled into ttnn by gdn_decode_step/sources.cmake
set(TTNN_OP_EXPERIMENTAL_GDN_SPEC_STEP_NANOBIND_SRCS gdn_spec_step_nanobind.cpp)
