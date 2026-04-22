// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms {

struct SubgroupGatherHistogramsParams {
    uint32_t cluster_axis;
    uint32_t num_dispatch_subgroups;
    uint32_t num_links;
    tt::tt_fabric::Topology topology;
    MemoryConfig output_mem_config;
    CoreRangeSet worker_core_range_set;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "cluster_axis",
        "num_dispatch_subgroups",
        "num_links",
        "topology",
        "output_mem_config",
        "worker_core_range_set");

    auto attribute_values() const {
        return std::forward_as_tuple(
            cluster_axis, num_dispatch_subgroups, num_links, topology, output_mem_config, worker_core_range_set);
    }
};

struct SubgroupGatherHistogramsInputs {
    ttnn::Tensor input_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms
