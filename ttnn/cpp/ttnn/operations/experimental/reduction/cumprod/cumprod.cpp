// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/cumprod_device_operation.hpp"
#include "cumprod.hpp"

namespace ttnn::operations::experimental::reduction {

Tensor CumprodOperation::invoke(
    const Tensor& input_tensor,
    const int32_t dim,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config,
    const QueueId& queue_id) {
    auto output_memory_config = optional_out.has_value() ? optional_out.value().memory_config()
                                                         : memory_config.value_or(input_tensor.memory_config());
    return ttnn::prim::cumprod(input_tensor, dim, optional_out, output_memory_config, queue_id);
}

}  // namespace ttnn::operations::experimental::reduction
