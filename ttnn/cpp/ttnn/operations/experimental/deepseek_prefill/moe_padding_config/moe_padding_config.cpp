// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_padding_config.hpp"

#include "device/moe_padding_config_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config {

ttnn::Tensor moe_padding_config(
    const ttnn::Tensor& config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& actual_end,
    uint32_t tokens_per_chip,
    uint32_t pad_side,
    uint32_t cluster_axis) {
    return ttnn::prim::moe_padding_config(config, actual_start, actual_end, tokens_per_chip, pad_side, cluster_axis);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config
