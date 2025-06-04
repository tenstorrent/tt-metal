// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::transformer {

struct ExecuteConcatenateHeads {
    static ttnn::Tensor invoke(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config);
};
}  // namespace operations::transformer

namespace transformer {
constexpr auto concatenate_heads = ttnn::register_operation<
    "ttnn::transformer::concatenate_heads",
    ttnn::operations::transformer::ExecuteConcatenateHeads>();

}  // namespace transformer
}  // namespace ttnn
