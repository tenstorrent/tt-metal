// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::pcie_exchange {

// mode 0 = all-reduce (sum), 1 = all-gather along the last dim. See device/pcie_exchange_device_operation_types.hpp.
Tensor pcie_exchange(
    const Tensor& input,
    const Tensor& exchange,
    uint32_t mode,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& ctrl = std::nullopt,
    bool bfp8_payload = false);

}  // namespace ttnn::operations::experimental::pcie_exchange
