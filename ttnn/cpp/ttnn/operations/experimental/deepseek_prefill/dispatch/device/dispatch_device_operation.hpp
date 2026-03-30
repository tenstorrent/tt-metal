// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <variant>

#include "dispatch_types.hpp"
#include "dispatch_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

namespace detail {

std::pair<std::array<uint32_t, 2>, std::array<uint32_t, 2>> get_cb_sizes(
    const Tensor& input_tensor,
    const Tensor& weights_tensor,
    const Tensor& indices_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis);

}  // namespace detail

struct DispatchDeviceOperation {
    using operation_attributes_t = DispatchParams;
    using tensor_args_t = DispatchInputs;
    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = std::array<Tensor, 2>;
    using program_factory_t = std::variant<DispatchProgramFactory>;

    // Mandatory methods
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch

namespace ttnn::prim {
ttnn::operations::experimental::deepseek_prefill::dispatch::DispatchDeviceOperation::tensor_return_value_t
prefill_dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weights_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    uint32_t dispatch_group_size,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatched_tokens_per_expert,
    std::optional<uint32_t> axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set);
}  // namespace ttnn::prim
