// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations {

namespace embedding_backward {

struct EmbeddingBackwardOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input_tensor_arg,
        const Tensor& weight_tensor_arg,
        const Tensor& output_gradient_tensor_arg,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor_arg,
        const Tensor& weight_tensor_arg,
        const Tensor& output_gradient_tensor_arg,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace embedding_backward
}  // namespace operations

constexpr auto embedding_bw =
    ttnn::register_operation<"ttnn::embedding_bw", ttnn::operations::embedding_backward::EmbeddingBackwardOperation>();

}  // namespace ttnn
