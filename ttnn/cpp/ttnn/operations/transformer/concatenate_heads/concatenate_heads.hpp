// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn::transformer {

/**
 * @brief Takes in a tensor of shape [batch_size, num_heads, sequence_size, head_size],
 * concatenates heads back along the width dimension and returns the tensor of shape
 * [batch_size, sequence_size, num_heads * head_size]
 */
ttnn::Tensor concatenate_heads(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::transformer
