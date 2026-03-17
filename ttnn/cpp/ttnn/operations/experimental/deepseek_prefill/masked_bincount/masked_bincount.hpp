// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental {

struct MaskedBincountOperation {
    static ttnn::Tensor invoke(const Tensor& input_tensor, const Tensor& expert_mask, uint32_t n_routed_experts);
};

}  // namespace operations::experimental

constexpr auto masked_bincount =
    ttnn::register_operation<"ttnn::masked_bincount", ttnn::operations::experimental::MaskedBincountOperation>();

}  // namespace ttnn
