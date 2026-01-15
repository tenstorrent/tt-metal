// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn {
namespace operations::experimental::transformer {
struct NLPConcatHeadsBoltzOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};
}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto nlp_concat_heads_boltz = ttnn::register_operation<
    "ttnn::experimental::nlp_concat_heads_boltz",
    ttnn::operations::experimental::transformer::NLPConcatHeadsBoltzOperation>();

}  // namespace experimental

}  // namespace ttnn
