set(TTNN_OP_EXPERIMENTAL_KDA_GDN_SPEC_TLOOP_PROTO_API_HEADERS gdn_spec_tloop_proto.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_GDN_SPEC_TLOOP_PROTO_SRCS
    gdn_spec_tloop_proto.cpp
    device/gdn_spec_tloop_proto_device_operation.cpp
    device/gdn_spec_tloop_proto_program_factory.cpp
)

# ../kda_nanobind.cpp is already compiled into ttnn by gdn_decode_step/sources.cmake
set(TTNN_OP_EXPERIMENTAL_KDA_GDN_SPEC_TLOOP_PROTO_NANOBIND_SRCS gdn_spec_tloop_proto_nanobind.cpp)
