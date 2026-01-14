// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn {
namespace operations::experimental::nlp_concat_heads {

struct NLPConcatHeadsOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt);
};
}  // namespace operations::experimental::nlp_concat_heads

namespace experimental {

constexpr auto nlp_concat_heads = ttnn::register_operation<
    "ttnn::experimental::nlp_concat_heads",
    ttnn::operations::experimental::nlp_concat_heads::NLPConcatHeadsOperation>();

}  // namespace experimental

}  // namespace ttnn
