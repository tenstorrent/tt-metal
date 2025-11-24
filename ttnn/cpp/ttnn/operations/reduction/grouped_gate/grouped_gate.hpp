// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction {

struct GroupedGateOperation {
    static std::tuple<Tensor, Tensor> invoke(
        const Tensor& scores,
        const Tensor& bias,
        const float route_scale,
        const float epsilon,
        const uint32_t n_groups,
        const uint32_t topk,
        const uint32_t topk_groups,
        const uint32_t n_activated_experts,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

}  // namespace ttnn::operations::reduction

namespace ttnn {
constexpr auto grouped_gate =
    ttnn::register_operation<"ttnn::grouped_gate", ttnn::operations::reduction::GroupedGateOperation>();
}  // namespace ttnn
