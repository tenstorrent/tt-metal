# Source files for ttnn_op_layer_send.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_LAYER_SEND_API_HEADERS layer_send.hpp)

set(TTNN_OP_LAYER_SEND_SRCS
    layer_send.cpp
    device/layer_send_device_operation.cpp
    device/layer_send_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/layer_send/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_LAYER_SEND_NANOBIND_SRCS ${CMAKE_CURRENT_LIST_DIR}/layer_send_nanobind.cpp)
