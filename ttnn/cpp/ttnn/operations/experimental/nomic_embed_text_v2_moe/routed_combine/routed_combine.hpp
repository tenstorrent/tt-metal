// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// The routed experts' result for each token: its top-k rows of the expert outputs, weighted and
// summed, plus the experts' shared bias. Without weights and bias, a one-slot gather of the rows,
// tilized. See the binding for the contract. Reads nothing back to the host.
Tensor routed_combine(
    const Tensor& expert_rows,
    const Tensor& combine_index,
    const std::optional<Tensor>& combine_weights,
    const std::optional<Tensor>& bias,
    uint32_t num_tokens,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    uint32_t batch = 1);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe
