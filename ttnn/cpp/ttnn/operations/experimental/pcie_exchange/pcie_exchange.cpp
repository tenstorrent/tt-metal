// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pcie_exchange.hpp"

#include "device/pcie_exchange_device_operation.hpp"

namespace ttnn::operations::experimental::pcie_exchange {

Tensor pcie_exchange(
    const Tensor& input,
    const Tensor& exchange,
    uint32_t mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& ctrl,
    bool bfp8_payload) {
    return ttnn::prim::pcie_exchange(
        input, exchange, static_cast<PcieExchangeMode>(mode), memory_config, ctrl, bfp8_payload);
}

}  // namespace ttnn::operations::experimental::pcie_exchange
