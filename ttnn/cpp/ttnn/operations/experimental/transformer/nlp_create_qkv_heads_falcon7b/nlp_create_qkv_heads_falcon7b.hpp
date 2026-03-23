// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_falcon7b_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_falcon7b(
    const Tensor& input_tensor_q, const std::optional<MemoryConfig>& memory_config);

}  // namespace ttnn::experimental
