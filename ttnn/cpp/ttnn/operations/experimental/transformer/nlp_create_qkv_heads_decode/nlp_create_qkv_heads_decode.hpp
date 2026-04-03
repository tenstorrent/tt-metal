// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_decode_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_decode(
    const Tensor& input_tensor,
    uint32_t num_heads,
    std::optional<const uint32_t> num_kv_heads,
    std::optional<std::array<Tensor, 3>>& optional_output_tensors,
    std::optional<const bool> overlap_qk_coregrid = true,
    const std::optional<const Tensor>& batch_offset = std::nullopt,
    std::optional<const uint32_t> slice_size = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
