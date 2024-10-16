// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nonzero.hpp"

#include "device/nonzero_device_operation.hpp"

namespace ttnn::operations {

std::vector<ttnn::Tensor> NonzeroOperation::invoke(
    const ttnn::Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::nonzero(input_tensor, memory_config);
}

}  // namespace ttnn::operations
