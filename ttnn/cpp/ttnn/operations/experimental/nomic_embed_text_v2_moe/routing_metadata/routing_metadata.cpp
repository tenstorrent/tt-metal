// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routing_metadata.hpp"

#include "device/routing_metadata_device_operation.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

std::tuple<Tensor, Tensor, Tensor, Tensor> routing_metadata(
    const Tensor& indices,
    const Tensor& values,
    uint32_t num_experts,
    uint32_t token_start,
    std::optional<uint32_t> num_tokens) {
    // Without num_tokens the window runs to the last token. A token_start at or past the end leaves
    // an empty window, which validation reports.
    const uint32_t total_tokens = indices.logical_shape().rank() >= 2 ? indices.logical_shape()[-2] : 0;
    const uint32_t window = num_tokens.value_or(total_tokens > token_start ? total_tokens - token_start : 0);
    return ttnn::prim::routing_metadata(indices, values, num_experts, token_start, window);
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
