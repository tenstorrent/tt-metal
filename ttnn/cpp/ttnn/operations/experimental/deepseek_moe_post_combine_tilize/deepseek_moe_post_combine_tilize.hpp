// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations::experimental::deepseek_moe_post_combine_tilize {

struct DeepseekMoEPostCombineTilizeOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config = std::nullopt);
};

}  // namespace operations::experimental::deepseek_moe_post_combine_tilize

namespace experimental::deepseek_moe_post_combine_tilize {

constexpr auto deepseek_moe_post_combine_tilize = ttnn::register_operation<
    "ttnn::experimental::deepseek_moe_post_combine_tilize",
    ttnn::operations::experimental::deepseek_moe_post_combine_tilize::DeepseekMoEPostCombineTilizeOperation>();

}  // namespace experimental::deepseek_moe_post_combine_tilize
}  // namespace ttnn
