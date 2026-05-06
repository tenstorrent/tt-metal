// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

struct DispatchParams {
    uint32_t dispatch_group_size;
    uint32_t experts_per_chip;
    uint32_t num_routed_experts;
    uint32_t num_experts_per_tok;
    uint32_t metadata_len;
    // Total token capacity of the dispatch buffer (shared across all local experts
    // via dynamic offsets). Used to size the per-chip dispatch buffer and as the
    // in-kernel bounds check.
    uint32_t max_dispatch_buffer_token_size;
    std::optional<uint32_t> axis;
    uint32_t num_links;
    tt::tt_fabric::Topology topology;
    MemoryConfig output_mem_config;
    CoreRangeSet worker_core_range_set;
    bool use_l1_small_for_semaphores = false;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dispatch_group_size",
        "experts_per_chip",
        "num_routed_experts",
        "num_experts_per_tok",
        "metadata_len",
        "max_dispatch_buffer_token_size",
        "axis",
        "num_links",
        "topology",
        "output_mem_config",
        "worker_core_range_set",
        "use_l1_small_for_semaphores");

    auto attribute_values() const {
        return std::forward_as_tuple(
            dispatch_group_size,
            experts_per_chip,
            num_routed_experts,
            num_experts_per_tok,
            metadata_len,
            max_dispatch_buffer_token_size,
            axis,
            num_links,
            topology,
            output_mem_config,
            worker_core_range_set,
            use_l1_small_for_semaphores);
    };
};

struct DispatchInputs {
    Tensor input_tensor;
    Tensor weights_tensor;
    Tensor indices_tensor;
    Tensor expert_offsets_tensor;
    Tensor expert_dispatch_table_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
