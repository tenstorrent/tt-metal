// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "selective_reduce_combine.hpp"
#include "device/selective_reduce_combine_device_operation.hpp"
// #include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl::moe {

ttnn::Tensor ExecuteSelectiveReduceCombine::invoke(
    const ttnn::Tensor& dense_input_tensor,
    const ttnn::Tensor& dense_metadata_tensor,
    const ttnn::Tensor& dense_token_maps_tensor,
    const ttnn::Tensor& dense_token_counts_tensor,
    const uint32_t hidden_size,
    const uint32_t batch_size,
    const uint32_t seq_size,
    const uint32_t select_experts_k,
    const uint32_t experts,
    const std::optional<uint32_t>& axis,
    tt::tt_fabric::Topology topology,
    const uint32_t num_links,
    const uint32_t num_token_parallel_cores,
    const uint32_t num_data_parallel_cores,
    const CoreRangeSet worker_core_range_set,
    const CoreRangeSet mux_core_range_set,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    auto input_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

    return ttnn::prim::selective_reduce_combine(
        dense_input_tensor,
        dense_metadata_tensor,
        dense_token_maps_tensor,
        dense_token_counts_tensor,
        hidden_size,
        batch_size,
        seq_size,
        select_experts_k,
        experts,
        axis,
        topology,
        num_links,
        num_token_parallel_cores,
        num_data_parallel_cores,
        worker_core_range_set,
        mux_core_range_set,
        input_memory_config,
        optional_output_tensor);
}

}  // namespace ttnn::operations::ccl::moe
