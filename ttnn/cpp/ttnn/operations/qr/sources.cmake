# Source files for ttnn_op_qr.

set(TTNN_OP_QR_API_HEADERS qr.hpp)

set(TTNN_OP_QR_SRCS
    qr.cpp
    device/qr_device_operation.cpp
    device/qr_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/qr/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_QR_NANOBIND_SRCS qr_nanobind.cpp)
