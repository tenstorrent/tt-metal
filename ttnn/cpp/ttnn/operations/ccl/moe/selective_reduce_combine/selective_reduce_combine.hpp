// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::ccl::moe {

struct ExecuteSelectiveReduceCombine {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& dense_input_tensor,
        const ttnn::Tensor& dense_metadata_tensor,
        const ttnn::Tensor& dense_token_maps_tensor,
        const ttnn::Tensor& dense_token_counts_tensor,
        uint32_t hidden_size,
        uint32_t batch_size,
        uint32_t seq_size,
        uint32_t select_experts_k,
        uint32_t experts,
        const std::optional<uint32_t>& axis,
        tt::tt_fabric::Topology topology,
        uint32_t num_links,
        uint32_t num_token_parallel_cores,
        uint32_t num_data_parallel_cores,
        const CoreRangeSet worker_core_range_set,
        const CoreRangeSet mux_core_range_set,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<GlobalSemaphore>& optional_cross_device_semaphore = std::nullopt);
};

}  // namespace operations::ccl::moe

constexpr auto selective_reduce_combine = ttnn::
    register_operation<"ttnn::selective_reduce_combine", ttnn::operations::ccl::moe::ExecuteSelectiveReduceCombine>();

}  // namespace ttnn
