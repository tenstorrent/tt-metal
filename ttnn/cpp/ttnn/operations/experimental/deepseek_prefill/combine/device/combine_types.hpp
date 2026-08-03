// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/global_semaphore.hpp>

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
    // Optional routed-expert global semaphore for overlapping the routed expert with the combine.
    // Only consumed by the reader_untilize kernel (waits on it before processing each expert).
    std::optional<tt::tt_metal::GlobalSemaphore> global_semaphore = std::nullopt;

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
        "global_semaphore");

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
            global_semaphore);
    };
};

struct CombineInputs {
    ttnn::Tensor dispatched_buffer;
    ttnn::Tensor dispatched_metadata;
    ttnn::Tensor expert_token_counts;
    ttnn::Tensor expert_region_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine
