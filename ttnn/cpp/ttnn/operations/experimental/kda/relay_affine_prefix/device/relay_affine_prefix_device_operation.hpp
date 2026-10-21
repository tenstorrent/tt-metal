// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include "ttnn/operation.hpp"
#include "relay_affine_prefix_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RelayAffinePrefixOperation {
    using operation_attributes_t = RelayAffinePrefixParams;
    using tensor_args_t = RelayAffinePrefixInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RelayAffinePrefixProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::tuple<Tensor, Tensor, std::optional<Tensor>> relay_affine_prefix(
    const Tensor& packed,
    const Tensor& initial_state,
    const Tensor& actual_start,
    const Tensor& entry_landing,
    const Tensor& state_staging,
    const Tensor& final_landing,
    const tt::tt_metal::MemoryConfig& entry_memory_config,
    const DeviceComputeKernelConfig&,
    const std::array<uint32_t, 2>& semaphore_addresses,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::optional<AffineRelayOutputGateParams>& output_gate,
    const std::optional<Tensor>& gate_input,
    const std::optional<Tensor>& gate_weight,
    const std::optional<Tensor>& gate_copy_source);

}  // namespace ttnn::experimental::prim
