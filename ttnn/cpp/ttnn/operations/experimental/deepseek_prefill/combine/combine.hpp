// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::experimental::deepseek_prefill::combine {

struct ExecuteCombine {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& dispatched_buffer,
        const ttnn::Tensor& dispatched_metadata,
        const ttnn::Tensor& expert_token_counts,
        uint32_t dispatch_group_size,
        uint32_t experts_per_chip,
        uint32_t num_experts_per_tok,
        uint32_t seq_len_per_chip,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        std::optional<uint32_t> cluster_axis = 0,
        std::optional<uint32_t> num_links = 1,
        std::optional<tt::tt_fabric::Topology> topology = tt::tt_fabric::Topology::Linear,
        bool init_zeros = true);
};

}  // namespace operations::experimental::deepseek_prefill::combine

constexpr auto combine = ttnn::register_operation<
    "ttnn::experimental::deepseek_prefill::combine",
    ttnn::operations::experimental::deepseek_prefill::combine::ExecuteCombine>();

}  // namespace ttnn
