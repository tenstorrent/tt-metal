set(TTNN_OP_BRINGUP_HIGH_BW_ALL_GATHER_TTNN_API_HEADERS high_bw_all_gather.hpp)

set(TTNN_OP_BRINGUP_HIGH_BW_ALL_GATHER_TTNN_SRCS
    device/high_bw_all_gather_device_operation.cpp
    device/high_bw_all_gather_fabric_factory.cpp
    high_bw_all_gather.cpp
)

set(TTNN_OP_BRINGUP_HIGH_BW_ALL_GATHER_TTNN_NANOBIND_SRCS high_bw_all_gather_nanobind.cpp)
