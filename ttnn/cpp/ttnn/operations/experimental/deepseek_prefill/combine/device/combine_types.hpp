// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine {

struct CombineParams {
    uint32_t dispatch_group_size;
    uint32_t experts_per_chip;
    uint32_t num_experts_per_tok;
    uint32_t seq_len_per_chip;
    std::optional<uint32_t> axis;
    uint32_t num_links;
    tt::tt_fabric::Topology topology;
    MemoryConfig output_mem_config;
    CoreRangeSet worker_core_range_set;
    bool init_zeros;
    bool use_l1_small_for_semaphores = false;
    bool use_fp8_combine = false;
    bool use_store_and_forward = false;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "dispatch_group_size",
        "experts_per_chip",
        "num_experts_per_tok",
        "seq_len_per_chip",
        "axis",
        "num_links",
        "topology",
        "output_mem_config",
        "worker_core_range_set",
        "init_zeros",
        "use_l1_small_for_semaphores",
        "use_fp8_combine",
        "use_store_and_forward");

    auto attribute_values() const {
        return std::forward_as_tuple(
            dispatch_group_size,
            experts_per_chip,
            num_experts_per_tok,
            seq_len_per_chip,
            axis,
            num_links,
            topology,
            output_mem_config,
            worker_core_range_set,
            init_zeros,
            use_l1_small_for_semaphores,
            use_fp8_combine,
            use_store_and_forward);
    };
};

struct CombineInputs {
    ttnn::Tensor dispatched_buffer;
    ttnn::Tensor dispatched_metadata;
    ttnn::Tensor expert_token_counts;
    ttnn::Tensor expert_region_offsets;
    // Caller-owned DRAM scratch for the store-and-forward path.  Required exactly when
    // use_store_and_forward is set and the mesh is deep enough for a relay to exist.  Never read
    // across invocations, so one buffer is shared by every layer rather than allocated per layer.
    std::optional<ttnn::Tensor> staging_buffer;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
