// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <variant>
#include <vector>

#include "dispatch_fabric2d_types.hpp"
#include "dispatch_fabric2d_program_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

struct DispatchFabric2dDeviceOperation {
    using operation_attributes_t = DispatchFabric2dParams;
    using tensor_args_t = DispatchFabric2dInputs;
    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 2>;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = std::array<ttnn::Tensor, 2>;
    using program_factory_t = std::variant<DispatchFabric2dProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d

namespace ttnn::prim {
std::array<ttnn::Tensor, 2> dispatch_fabric2d(
    ttnn::MeshDevice* device,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& expert_offsets_tensor,
    const ttnn::Tensor& expert_dispatch_table_tensor,
    uint32_t experts_per_chip,
    uint32_t num_routed_experts,
    uint32_t num_experts_per_tok,
    uint32_t metadata_len,
    uint32_t max_dispatch_buffer_token_size,
    uint32_t seq_len_per_chip,
    uint32_t axis,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const tt::tt_metal::MemoryConfig& memory_config);
}  // namespace ttnn::prim
