// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// An MoE layer's router front end in one pass: each token's probabilities over the experts, softmax(input @
// weight) in fp32, and the input's rows row-major for the routed dispatch. See the binding for the contract.
// Reads nothing back to the host.
std::tuple<Tensor, Tensor> router_front_end(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<tt::tt_metal::MemoryConfig>& scores_memory_config = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& rows_memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
