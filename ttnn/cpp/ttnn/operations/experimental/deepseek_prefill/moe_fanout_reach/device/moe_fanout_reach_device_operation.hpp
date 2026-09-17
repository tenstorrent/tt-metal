// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "moe_fanout_reach_device_operation_types.hpp"
#include "moe_fanout_reach_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach {

struct MoeFanoutReachDeviceOperation {
    using operation_attributes_t = MoeFanoutReachParams;
    using tensor_args_t = MoeFanoutReachInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using topology_return_value_t = tt::tt_metal::TensorTopology;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<MoeFanoutReachProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach

namespace ttnn::prim {
ttnn::Tensor moe_fanout_reach(
    ttnn::MeshDevice* device,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    const ttnn::Tensor& global_dispatch_offsets,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t dispatch_group_size,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t axis,
    const tt::tt_metal::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set);
}  // namespace ttnn::prim
