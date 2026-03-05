// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::experimental::deepseek::prefill_dispatch {

struct ExecutePrefillDispatch {
    static std::array<ttnn::Tensor, 2> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& weights_tensor,
        const ttnn::Tensor& indices_tensor,
        const ttnn::Tensor& chip_to_n_routed_expert_offset_tensor,
        uint32_t num_chips,
        uint32_t experts_per_chip,
        uint32_t n_routed_experts,
        uint32_t num_experts_per_tok,
        uint32_t metadata_len,
        uint32_t max_dispatched_tokens_per_expert,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = 0,
        std::optional<uint32_t> num_links = 1,
        std::optional<tt::tt_fabric::Topology> topology = tt::tt_fabric::Topology::Linear);
};

}  // namespace operations::experimental::deepseek::prefill_dispatch

constexpr auto prefill_dispatch = ttnn::register_operation<
    "ttnn::experimental::deepseek::prefill_dispatch",
    ttnn::operations::experimental::deepseek::prefill_dispatch::ExecutePrefillDispatch>();

}  // namespace ttnn
