// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct NLPConcatHeadsDecodeOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor,
        uint32_t num_heads,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        uint32_t num_heads,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};
}  // namespace operations::experimental::transformer

namespace experimental {

constexpr auto nlp_concat_heads_decode = ttnn::register_operation<
    "ttnn::experimental::nlp_concat_heads_decode",
    ttnn::operations::experimental::transformer::NLPConcatHeadsDecodeOperation>();

}  // namespace experimental

}  // namespace ttnn
