// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental {

struct GeluBackwardOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& grad_output_tensor,
        const Tensor& input_tensor,
        const string& approximate,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> input_grad_tensor = std::nullopt);
};

}  // namespace ttnn::operations::experimental

namespace ttnn::experimental {
constexpr auto gelu_bw =
    ttnn::register_operation<"ttnn::experimental::gelu_bw", ttnn::operations::experimental::GeluBackwardOperation>();
}
