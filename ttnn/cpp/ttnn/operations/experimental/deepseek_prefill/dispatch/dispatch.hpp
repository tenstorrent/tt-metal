// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

std::array<ttnn::Tensor, 2> dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weights_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    std::optional<uint32_t> cluster_axis = 0,
    std::optional<uint32_t> num_links = 1,
    std::optional<tt::tt_fabric::Topology> topology = tt::tt_fabric::Topology::Linear);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch

namespace ttnn {
using operations::experimental::deepseek_prefill::dispatch::dispatch;
}  // namespace ttnn
