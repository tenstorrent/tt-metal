// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::moe {

struct ExecuteMoE {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& w0_w1_tensor,
        const ttnn::Tensor& w2_tensor,
        const ttnn::Tensor& output_tensor,
        const uint32_t num_experts,
        const uint32_t layer_id,
        const tt::tt_metal::CoreRangeSet& output_shard_core_ranges);
};

}  // namespace ttnn::operations::experimental::moe

namespace ttnn::experimental {
constexpr auto moe =
    ttnn::register_operation<"ttnn::experimental::moe", ttnn::operations::experimental::moe::ExecuteMoE>();
}  // namespace ttnn::experimental
