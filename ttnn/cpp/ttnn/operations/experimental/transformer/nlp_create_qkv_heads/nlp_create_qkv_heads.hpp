// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/nlp_create_qkv_heads_device_operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// ``kv_tied``: the fused input carries Q and a single K/V section -- width
// (num_q_heads + num_kv_heads) * head_dim -- and V is read from the same columns as K.
// For models that tie K and V to one projection (Gemma4's global layers, where V is
// v_norm(kv) and K is rope(k_norm(kv)), so the two differ only downstream of this op),
// this keeps the duplicate columns out of the projection matmul. K and V outputs are
// still two distinct tensors, as the callers immediately diverge on them.
std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors = std::nullopt,
    bool kv_tied = false);

}  // namespace ttnn::experimental
