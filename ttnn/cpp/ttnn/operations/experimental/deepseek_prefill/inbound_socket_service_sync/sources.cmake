# Source files for ttnn_op_experimental_deepseek_prefill_inbound_socket_service_sync.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_H2D_SOCKET_SYNC_SRCS
    device/inbound_socket_service_sync_device_operation.cpp
    device/inbound_socket_service_sync_program_factory.cpp
    inbound_socket_service_sync.cpp
)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_H2D_SOCKET_SYNC_API_HEADERS inbound_socket_service_sync.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/inbound_socket_service_sync/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_H2D_SOCKET_SYNC_NANOBIND_SRCS inbound_socket_service_sync_nanobind.cpp)
