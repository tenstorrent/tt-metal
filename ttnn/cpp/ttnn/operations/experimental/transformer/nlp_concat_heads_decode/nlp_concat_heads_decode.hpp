// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::nlp_concat_heads_decode {

struct NLPConcatHeadsDecodeOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        uint32_t num_heads,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};
}  // namespace operations::experimental::nlp_concat_heads_decode

namespace experimental {

constexpr auto nlp_concat_heads_decode = ttnn::register_operation<
    "ttnn::experimental::nlp_concat_heads_decode",
    ttnn::operations::experimental::nlp_concat_heads_decode::NLPConcatHeadsDecodeOperation>();

}  // namespace experimental

}  // namespace ttnn
