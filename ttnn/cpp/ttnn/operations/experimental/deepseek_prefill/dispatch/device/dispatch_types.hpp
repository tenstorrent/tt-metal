// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    uint32_t max_dispatched_tokens_per_expert;
    std::optional<uint32_t> axis;
    uint32_t num_links;
    tt::tt_fabric::Topology topology;
    MemoryConfig output_mem_config;
    CoreRangeSet worker_core_range_set;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dispatch_group_size",
        "experts_per_chip",
        "num_routed_experts",
        "num_experts_per_tok",
        "metadata_len",
        "max_dispatched_tokens_per_expert",
        "axis",
        "num_links",
        "topology",
        "output_mem_config",
        "worker_core_range_set");

    auto attribute_values() const {
        return std::forward_as_tuple(
            dispatch_group_size,
            experts_per_chip,
            num_routed_experts,
            num_experts_per_tok,
            metadata_len,
            max_dispatched_tokens_per_expert,
            axis,
            num_links,
            topology,
            output_mem_config,
            worker_core_range_set);
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
