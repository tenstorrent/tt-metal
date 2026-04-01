// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_segformer(
    const Tensor& input_tensor_q,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt);

}  // namespace ttnn::experimental
