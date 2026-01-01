// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// From a top-k router's selection, builds what a routed MoE pass needs to send each token only to
// its own experts: gather_index, combine_index, combine_weights and expert_counts. See the binding
// for the contract. Reads nothing back to the host.
std::tuple<Tensor, Tensor, Tensor, Tensor> routing_metadata(
    const Tensor& indices,
    const Tensor& values,
    uint32_t num_experts,
    uint32_t token_start = 0,
    std::optional<uint32_t> num_tokens = std::nullopt);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
