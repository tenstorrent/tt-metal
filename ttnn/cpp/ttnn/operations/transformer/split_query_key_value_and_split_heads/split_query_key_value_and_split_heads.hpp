// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::transformer {}  // namespace operations::transformer

namespace transformer {

/**
 * @brief Splits input_tensor of shape [batch_size, sequence_size, 3 * hidden_size] into 3 tensors (Query, Key, Value)
 * of shape [batch_size, sequence_size, hidden_size]. Then, reshapes and permutes the output tensors, to make them 
 * ready for computing attention scores.
 *
 * If kv_input_tensor is passed in, then input_tensor of shape [batch_size, sequence_size, hidden_size] is only used 
 * for Query, and kv_input_tensor of shape [batch_size, sequence_size, 2 * hidden_size] is used for Key and Value.
 */
std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_heads,
    std::optional<uint32_t> num_kv_heads,
    bool transpose_key,
    const std::optional<MemoryConfig>& memory_config);

}  // namespace transformer
}  // namespace ttnn
