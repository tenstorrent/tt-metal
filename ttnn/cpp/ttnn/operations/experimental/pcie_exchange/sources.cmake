set(TTNN_OP_EXPERIMENTAL_PCIE_EXCHANGE_API_HEADERS pcie_exchange.hpp)

set(TTNN_OP_EXPERIMENTAL_PCIE_EXCHANGE_SRCS
    device/pcie_exchange_device_operation.cpp
    device/pcie_exchange_program_factory.cpp
    pcie_exchange.cpp
)

set(TTNN_OP_EXPERIMENTAL_PCIE_EXCHANGE_NANOBIND_SRCS pcie_exchange_nanobind.cpp)
