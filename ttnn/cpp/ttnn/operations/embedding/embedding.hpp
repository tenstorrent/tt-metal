// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/embedding_device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::embedding {

struct EmbeddingOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor_arg,
        const Tensor& weight_arg,
        const std::optional<int>& pad_token = std::nullopt,
        const std::optional<Layout>& layout = std::nullopt,
        ttnn::prim::EmbeddingsType embeddings_type = ttnn::prim::EmbeddingsType::GENERIC,
        std::optional<const DataType> dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace operations::embedding

constexpr auto embedding =
    ttnn::register_operation<"ttnn::embedding", ttnn::operations::embedding::EmbeddingOperation>();

}  // namespace ttnn
