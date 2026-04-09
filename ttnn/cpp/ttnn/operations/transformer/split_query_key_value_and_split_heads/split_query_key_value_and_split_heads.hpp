// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ttnn/types.hpp"

namespace ttnn::transformer {

/// QKV input layout for split_query_key_value_and_split_heads.
///
/// Two layouts exist in the framework. Until #41718 they were an implicit,
/// kernel-comment-only convention; this enum makes them part of the API so the
/// caller can declare their intent and the op can validate.
///
/// CONCATENATED — what nn.Linear(d, 3d) produces by default. Per (batch, seq) row:
///   [Q_h0, Q_h1, ..., Q_h(n_q-1), K_h0, ..., K_h(n_kv-1), V_h0, ..., V_hn]
///
/// GROUPED — GQA-packed layout the existing sharded
/// experimental::create_qkv_heads reader expects. Per (batch, seq) row, with
/// n = n_q / n_kv Q heads per KV group:
///   [Q_g0_h0, ..., Q_g0_h(n-1), K_g0, V_g0, Q_g1_h0, ..., K_g1, V_g1, ...]
///
/// AUTO — infer from memory config: GROUPED for sharded, CONCATENATED for
/// interleaved (matches the pre-#41718 implicit convention; existing callers
/// do not need to change).
///
/// SD U-Net cross-attention and ViT WH (the only two production sharded callers)
/// both manually repack their nn.Linear weights into GROUPED layout to match the
/// kernel's expectation — see concatenate_qkv() in stable_diffusion/wormhole/tt/
/// ttnn_functional_cross_attention.py:81-117 and the query_key_value weight
/// construction in vit/wormhole/tt/ttnn_optimized_sharded_vit_wh.py:559-566.
enum class QkvLayout : uint8_t {
    AUTO,
    CONCATENATED,
    GROUPED,
};

/**
 * @brief Splits input_tensor of shape [batch_size, sequence_size, 3 * hidden_size] into 3 tensors (Query, Key, Value)
 * of shape [batch_size, sequence_size, hidden_size]. Then, reshapes and permutes the output tensors, to make them
 * ready for computing attention scores.
 *
 * If kv_input_tensor is passed in, then input_tensor of shape [batch_size, sequence_size, hidden_size] is only used
 * for Query, and kv_input_tensor of shape [batch_size, sequence_size, 2 * hidden_size] is used for Key and Value.
 *
 * @param qkv_layout Explicit input layout. AUTO (the default) infers GROUPED for sharded inputs and
 *   CONCATENATED for interleaved inputs (matches the pre-#41718 implicit convention; existing callers do not
 *   need to change). Pass CONCATENATED or GROUPED explicitly to make the layout convention part of the call
 *   site instead of relying on inference. Mismatched combinations (e.g. CONCATENATED + sharded, the cause of
 *   the original bug from #41526) are rejected with a clear TT_FATAL.
 */
std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(
    const Tensor& input_tensor,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_heads,
    std::optional<uint32_t> num_kv_heads,
    bool transpose_key,
    const std::optional<MemoryConfig>& memory_config,
    bool use_falcon7b_backend = false,
    QkvLayout qkv_layout = QkvLayout::AUTO);

}  // namespace ttnn::transformer
