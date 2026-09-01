set(TTNN_OP_EXPERIMENTAL_KDA_AFFINE_EXCLUSIVE_SCAN_API_HEADERS affine_exclusive_scan.hpp)

set(TTNN_OP_EXPERIMENTAL_KDA_AFFINE_EXCLUSIVE_SCAN_SRCS
    affine_exclusive_scan.cpp
    device/affine_exclusive_scan_device_operation.cpp
    device/affine_exclusive_scan_program_factory.cpp
)

set(TTNN_OP_EXPERIMENTAL_KDA_AFFINE_EXCLUSIVE_SCAN_NANOBIND_SRCS
    affine_exclusive_scan_nanobind.cpp
    ../kda_nanobind.cpp
)
