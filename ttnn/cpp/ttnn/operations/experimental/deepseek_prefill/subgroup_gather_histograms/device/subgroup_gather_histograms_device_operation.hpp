// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "subgroup_gather_histograms_device_operation_types.hpp"
#include "subgroup_gather_histograms_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms {

struct SubgroupGatherHistogramsDeviceOperation {
    using operation_attributes_t = SubgroupGatherHistogramsParams;
    using tensor_args_t = SubgroupGatherHistogramsInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<SubgroupGatherHistogramsProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::subgroup_gather_histograms

namespace ttnn::prim {
ttnn::Tensor subgroup_gather_histograms(
    const ttnn::Tensor& input_tensor,
    uint32_t cluster_axis,
    uint32_t num_dispatch_subgroups,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set);
}  // namespace ttnn::prim
