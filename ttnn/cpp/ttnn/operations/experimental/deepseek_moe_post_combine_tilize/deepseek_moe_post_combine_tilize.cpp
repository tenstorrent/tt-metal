// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <optional>

#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/deepseek_moe_post_combine_tilize.hpp"
#include "ttnn/operations/experimental/deepseek_moe_post_combine_tilize/device/deepseek_moe_post_combine_tilize_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

ttnn::Tensor deepseek_moe_post_combine_tilize(
    const ttnn::Tensor& input_tensor, const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    return ttnn::prim::deepseek_moe_post_combine_tilize(input_tensor, output_memory_config);
}

}  // namespace ttnn::experimental
