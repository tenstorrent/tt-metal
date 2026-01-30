// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteMoE {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& tilize_input_tensor,
        const ttnn::Tensor& tilize_expert_indices_tensor,
        const ttnn::Tensor& tilize_expert_scores_tensor,
        const ttnn::Tensor& tilize_expert_mapping_tensor,
        const ttnn::Tensor& matmul_w0_w1_tensor,
        const ttnn::Tensor& matmul_w2_tensor,
        const uint32_t layer_id,
        const std::optional<uint32_t> cluster_axis);
};

}  // namespace operations::experimental::ccl

namespace experimental {

constexpr auto moe =
    ttnn::register_operation<"ttnn::experimental::moe", ttnn::operations::experimental::ccl::ExecuteMoE>();

}  // namespace experimental
}  // namespace ttnn
